"""
FastAPI 서버 - v17 Production (Hugging Face Hub)
"""

from fastapi import FastAPI, HTTPException
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
    """사기 여부 예측"""

    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="모델 로드 안됨")

    try:
        # 1. 전처리
        df = pd.DataFrame([job_data])
        preprocessor = model_artifacts['preprocessor']
        df = preprocessor.preprocess(df)

        # 2. Domain features
        extractor = model_artifacts['feature_extractor']
        features = extractor.extract_all_features(df.iloc[0].to_dict())
        X_domain = pd.DataFrame([features]).fillna(0).replace([np.inf, -np.inf], 0)

        # 3. TF-IDF
        tfidf = model_artifacts['tfidf']
        text = df['title'].fillna('').iloc[0] + ' ' + df['description'].fillna('').iloc[0]
        X_tfidf = tfidf.transform([text]).toarray()

        # 4. FastText (있으면)
        embedder = model_artifacts.get('embedder')
        if embedder and hasattr(embedder, 'model') and embedder.model:
            X_ft = embedder.transform([text])
            X = np.hstack([X_domain.values, X_tfidf, X_ft])
        else:
            X = np.hstack([X_domain.values, X_tfidf])

        # 5. 예측
        model = model_artifacts['model']
        threshold = model_artifacts.get('threshold', 0.08)

        proba = model.predict_proba(X)[0, 1]
        is_fraud = bool(proba >= threshold)

        # 6. Confidence (threshold 기준)
        if is_fraud:
            confidence = min((proba - threshold) / (1 - threshold), 1.0)
        else:
            confidence = min((threshold - proba) / threshold, 1.0)

        # 7. Risk level
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
            'model_version': model_artifacts.get('version', 'v17')
        }

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
        logger.info("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning("⚠️  모델 로드 실패 (헬스체크만 가능)")
        logger.warning("=" * 70)

    yield

    # Shutdown
    logger.info("서버 종료...")


app = FastAPI(
    title="사기 탐지 API v17",
    version="17.0",
    description="Fake Job Posting Detector - Powered by Hugging Face",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "사기 탐지 API v17",
        "status": "running",
        "model_loaded": model_artifacts is not None,
        "model_info": {
            "version": model_artifacts.get('version') if model_artifacts else None,
            "threshold": model_artifacts.get('threshold') if model_artifacts else None,
            "source": "https://huggingface.co/functionss/fake-job-detector"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    if model_artifacts:
        metrics = model_artifacts.get('metrics', {})
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": model_artifacts.get('version'),
            "threshold": model_artifacts.get('threshold'),
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
        - model_version: Model version
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
