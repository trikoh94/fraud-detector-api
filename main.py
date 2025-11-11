"""
FastAPI 서버 - v17 Production
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import requests
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="사기 탐지 API v17", version="17.0")

MODEL_PATH = Path("/app/fraud_detector_v17_recall75.joblib")
MODEL_DOWNLOAD_URL = os.getenv("MODEL_URL", "")  # 환경변수로!
model_artifacts = None


# ============================================================================
# 모델 다운로드
# ============================================================================

def download_model():
    """Google Drive에서 모델 다운로드"""
    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size
        if file_size > 1_000_000:
            logger.info(f"기존 모델 사용: {file_size / (1024*1024):.2f} MB")
            return True

    if not MODEL_DOWNLOAD_URL:
        logger.warning("MODEL_URL 환경변수 없음")
        return False

    logger.info("모델 다운로드 시작...")

    try:
        response = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=300)
        response.raise_for_status()

        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"다운로드 완료: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
        return True

    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        return False


# ============================================================================
# 모델 로드
# ============================================================================

def load_model():
    """모델 로드"""
    global model_artifacts

    logger.info("모델 로딩 시작...")

    if not download_model():
        return None

    try:
        artifacts = joblib.load(MODEL_PATH)

        logger.info("모델 로드 완료!")
        logger.info(f"  버전: {artifacts.get('version', 'unknown')}")
        logger.info(f"  Threshold: {artifacts.get('threshold', 0.20):.3f}")

        if 'metrics' in artifacts:
            metrics = artifacts['metrics']
            logger.info(f"  Recall: {metrics.get('recall', 0):.2%}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.2%}")
            logger.info(f"  F1: {metrics.get('f1', 0):.2%}")

        return artifacts

    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


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
        threshold = model_artifacts.get('threshold', 0.20)

        proba = model.predict_proba(X)[0, 1]
        is_fraud = bool(proba >= threshold)

        # 6. Confidence (threshold 기준)
        if is_fraud:
            confidence = min((proba - threshold) / (1 - threshold), 1.0)
        else:
            confidence = min((threshold - proba) / threshold, 1.0)

        # 7. Risk level (threshold 기준)
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
# Pydantic
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
    is_fraud: bool
    fraud_probability: float
    confidence: float
    risk_level: str
    threshold: float
    model_version: str


# ============================================================================
# API
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global model_artifacts
    logger.info("서버 시작...")
    model_artifacts = load_model()

    if model_artifacts:
        logger.info("서버 준비 완료!")
    else:
        logger.warning("모델 없이 시작 (헬스체크만)")


@app.get("/")
async def root():
    return {
        "message": "사기 탐지 API v17",
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
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "model_version": model_artifacts.get('version') if model_artifacts else None,
        "threshold": model_artifacts.get('threshold') if model_artifacts else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(job: JobPosting):
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="모델 로드 안됨")

    try:
        job_dict = job.model_dump()
        result = predict_fraud(job_dict)

        logger.info(f"예측 완료: {result['is_fraud']} ({result['fraud_probability']:.2%})")

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")