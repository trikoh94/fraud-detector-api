"""
FastAPI 서버 - v17 Lightweight 모델
수정: dill → joblib
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # ✅ dill 대신 joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="사기 탐지 API v17", version="17.0")

# 전역 변수
MODEL_PATH = Path("/app/model_v17_lightweight.pkl")
model_artifacts = None


# ============================================================================
# Pydantic 모델
# ============================================================================

class JobPosting(BaseModel):
    """채용 공고 입력"""
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
    """예측 결과"""
    is_fraud: bool
    fraud_probability: float
    confidence: float
    risk_level: str
    model_version: str


# ============================================================================
# 모델 로드
# ============================================================================

def load_model():
    """모델 로드 (joblib)"""
    global model_artifacts

    logger.info("🔄 모델 로딩 시작...")

    if not MODEL_PATH.exists():
        logger.warning(f"⚠️  모델 파일 없음: {MODEL_PATH}")
        return None

    logger.info(f"✅ 모델 발견: {MODEL_PATH.name}")

    try:
        logger.info(f"📂 로딩 중: {MODEL_PATH}")

        # ✅ joblib로 로드
        artifacts = joblib.load(MODEL_PATH)

        logger.info("✅ 모델 로드 완료!")
        logger.info(f"   버전: {artifacts.get('version', 'unknown')}")
        logger.info(f"   Threshold: {artifacts.get('threshold', 0.11):.3f}")

        if 'metrics' in artifacts:
            metrics = artifacts['metrics']
            logger.info(f"   Recall: {metrics.get('recall', 0):.2%}")
            logger.info(f"   Precision: {metrics.get('precision', 0):.2%}")

        return artifacts

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# 예측 함수
# ============================================================================

def predict_fraud(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """사기 여부 예측"""

    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    try:
        # 1. DataFrame 변환
        df = pd.DataFrame([job_data])

        # 2. 전처리
        preprocessor = model_artifacts['preprocessor']
        df = preprocessor.preprocess(df)

        # 3. Feature 추출
        extractor = model_artifacts['feature_extractor']
        features = extractor.extract_all_features(df.iloc[0].to_dict())

        X_domain = pd.DataFrame([features])
        X_domain = X_domain.fillna(0)
        X_domain = X_domain.replace([np.inf, -np.inf], 0)

        # 4. TF-IDF
        tfidf = model_artifacts['tfidf']
        text = df['title'].fillna('').iloc[0] + ' ' + df['description'].fillna('').iloc[0]
        X_tfidf = tfidf.transform([text]).toarray()

        # 5. 결합
        X = np.hstack([X_domain.values, X_tfidf])

        # 6. 예측
        model = model_artifacts['model']
        threshold = model_artifacts['threshold']

        proba = model.predict_proba(X)[0, 1]
        is_fraud = bool(proba >= threshold)
        confidence = float(abs(proba - 0.5) * 2)

        # 7. 위험도
        if proba >= 0.8:
            risk_level = "매우 높음"
        elif proba >= 0.5:
            risk_level = "높음"
        elif proba >= 0.3:
            risk_level = "중간"
        elif proba >= 0.15:
            risk_level = "낮음"
        else:
            risk_level = "매우 낮음"

        return {
            'is_fraud': is_fraud,
            'fraud_probability': float(proba),
            'confidence': confidence,
            'risk_level': risk_level,
            'model_version': model_artifacts.get('version', 'v17_lightweight')
        }

    except Exception as e:
        logger.error(f"❌ 예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {str(e)}")


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global model_artifacts
    logger.info("🚀 서버 시작...")
    model_artifacts = load_model()

    if model_artifacts is None:
        logger.warning("⚠️  모델 없이 서버 시작 (헬스체크만 가능)")
    else:
        logger.info("✅ 서버 준비 완료!")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "사기 탐지 API v17 Lightweight",
        "status": "running",
        "model_loaded": model_artifacts is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "model_version": model_artifacts.get('version', 'unknown') if model_artifacts else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(job: JobPosting):
    """채용 공고 사기 여부 예측"""

    if model_artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. 잠시 후 다시 시도해주세요."
        )

    try:
        # Pydantic → dict
        job_dict = job.model_dump()

        # 예측
        result = predict_fraud(job_dict)

        logger.info(f"✅ 예측 완료: {result['is_fraud']} ({result['fraud_probability']:.2%})")

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 예측 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 서버 시작: 0.0.0.0:8080")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )