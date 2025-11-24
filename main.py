"""
FastAPI 서버 - v33 Production (Hugging Face Hub)
FastText Optimized (min_count=3, vocab=15k)
CORS enabled for Chrome Extensions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ CORS 추가
from pydantic import BaseModel
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from model import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global Variables
# ============================================================================
model_artifacts = None


# ============================================================================
# 예측
# ============================================================================

def predict_fraud(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """사기 여부 예측 - v33 최적화"""

    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="모델 로드 안됨")

    try:
        # 1. 전처리
        df = pd.DataFrame([job_data])
        preprocessor = model_artifacts.get('preprocessor')

        if preprocessor:
            df = preprocessor.preprocess(df)
            job_dict = df.iloc[0].to_dict()
        else:
            # preprocessor 없으면 기본 정리만
            job_dict = job_data

        # 2. Domain features (83개)
        extractor = model_artifacts['feature_extractor']
        feature_columns = model_artifacts['feature_columns']

        domain_features = extractor.extract_all_features(job_dict)
        domain_df = pd.DataFrame([domain_features]).fillna(0).replace([np.inf, -np.inf], 0)
        domain_df = domain_df.reindex(columns=feature_columns, fill_value=0)
        X_domain = domain_df.values

        # 3. TF-IDF (500개)
        tfidf = model_artifacts['tfidf']
        title = str(job_dict.get('title', ''))
        description = str(job_dict.get('description', ''))
        text = f"{title} {description}"
        X_tfidf = tfidf.transform([text]).toarray()

        # 4. FastText (100개) - v33 Dictionary 방식!
        fasttext_embedder = model_artifacts['fasttext_embedder']
        X_fasttext = fasttext_embedder.get_embedding(text).reshape(1, -1)

        # 5. Feature 결합 (683개)
        X_final = np.hstack([X_domain, X_tfidf, X_fasttext])

        # Feature 개수 검증
        expected_features = len(feature_columns) + tfidf.max_features + fasttext_embedder.vector_size
        if X_final.shape[1] != expected_features:
            logger.error(f"Feature mismatch: expected {expected_features}, got {X_final.shape[1]}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature shape mismatch: expected {expected_features}, got {X_final.shape[1]}"
            )

        # 6. 예측
        model = model_artifacts['model']
        threshold = model_artifacts.get('threshold', 0.045)

        proba = model.predict_proba(X_final)[0, 1]
        is_fraud = bool(proba >= threshold)

        # 7. Confidence (threshold 기준)
        if is_fraud:
            confidence = min((proba - threshold) / (1 - threshold), 1.0)
        else:
            confidence = min((threshold - proba) / threshold, 1.0)

        # 8. Risk level
        if proba >= threshold * 2:
            risk_level = "매우 높음"
        elif proba >= threshold * 1.5:
            risk_level = "높음"
        elif proba >= threshold:
            risk_level = "중간"
        elif proba >= threshold * 0.5:
            risk_level = "낮음"
        else:
            risk_level = "매우 낮음"

        return {
            'is_fraud': is_fraud,
            'fraud_probability': float(proba),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'threshold': float(threshold),
            'model_version': model_artifacts.get('version', 'v33')
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"예측 오류: {str(e)}")


# ============================================================================
# Pydantic Models
# ============================================================================

class JobPosting(BaseModel):
    title: str
    location: Optional[str] = ""
    department: Optional[str] = ""
    salary_range: Optional[str] = ""
    company_profile: Optional[str] = ""
    description: str
    requirements: Optional[str] = ""
    benefits: Optional[str] = ""
    telecommuting: Optional[int] = 0
    has_company_logo: Optional[int] = 0
    has_questions: Optional[int] = 0
    employment_type: Optional[str] = ""
    required_experience: Optional[str] = ""
    required_education: Optional[str] = ""
    industry: Optional[str] = ""
    function: Optional[str] = ""


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    is_fraud: bool
    fraud_probability: float
    confidence: float
    risk_level: str
    threshold: float
    model_version: str


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # Startup
    global model_artifacts
    logger.info("🚀 서버 시작...")
    logger.info("=" * 70)

    model_artifacts = load_model()

    if model_artifacts:
        logger.info("=" * 70)
        logger.info("✅ 서버 준비 완료!")
        logger.info(f"📊 Model Version: {model_artifacts.get('version', 'v33')}")
        logger.info(f"🎯 Threshold: {model_artifacts.get('threshold', 0.045):.4f}")
        logger.info(f"🔤 FastText Vocab: {model_artifacts['fasttext_embedder'].vocab_size:,}")
        logger.info(f"📊 Features: 683 (83 domain + 500 TF-IDF + 100 FastText)")
        logger.info("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning("⚠️  모델 로드 실패 (헬스체크만 가능)")
        logger.warning("=" * 70)

    yield

    # Shutdown
    logger.info("서버 종료...")


app = FastAPI(
    title="사기 탐지 API v33",
    version="33.0",
    description="Fake Job Posting Detector - FastText Optimized (Powered by Hugging Face)",
    lifespan=lifespan
)

# 🔥 CORS 설정 추가 (Chrome Extension 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 Origin 허용 (프로덕션에서는 특정 도메인만 허용 권장)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],  # OPTIONS, HEAD 명시적 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"]
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "사기 탐지 API v33 - FastText Optimized",
        "status": "running",
        "model_loaded": model_artifacts is not None,
        "model_info": {
            "version": model_artifacts.get('version') if model_artifacts else None,
            "threshold": model_artifacts.get('threshold') if model_artifacts else None,
            "fasttext_vocab": model_artifacts['fasttext_embedder'].vocab_size if model_artifacts else None,
            "features": "683 (83 domain + 500 TF-IDF + 100 FastText)",
            "optimization": "min_count=3, vocab=15k",
            "source": "https://huggingface.co/functionss/fruaddetectionv2"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - 서버 상태 확인"""
    if model_artifacts:
        metrics = model_artifacts.get('metrics', {})
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": model_artifacts.get('version'),
            "threshold": model_artifacts.get('threshold'),
            "fasttext_vocab": model_artifacts['fasttext_embedder'].vocab_size,
            "total_features": 683,
            "performance": {
                "recall": metrics.get('recall'),
                "precision": metrics.get('precision'),
                "f1": metrics.get('f1'),
                "roc_auc": metrics.get('roc_auc')
            }
        }
    else:
        return {
            "status": "degraded",
            "model_loaded": False,
            "message": "Model not loaded"
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(job: JobPosting):
    """
    Predict if a job posting is fraudulent

    Returns:
        - is_fraud: Boolean indicating if job is fraudulent
        - fraud_probability: Probability score (0-1)
        - confidence: Model confidence (0-1)
        - risk_level: Risk assessment (매우 낮음 ~ 매우 높음)
        - threshold: Decision threshold used
        - model_version: Model version (v33)
    """

    if model_artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        job_dict = job.model_dump()
        result = predict_fraud(job_dict)

        logger.info(
            f"예측: {result['is_fraud']} "
            f"({result['fraud_probability']:.2%}) - "
            f"{job_dict.get('title', 'N/A')[:50]}"
        )

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )