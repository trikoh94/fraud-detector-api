"""
LinkedIn Fraud Detector API - Railway 배포용 (BERT 제거)
✅ BERT 완전 제거 (메모리 최적화)
✅ v17 모델 파일명 자동 호환
✅ Railway 최적화 완료
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import logging

# ========================================
# numpy 호환성 패치 (필수!)
# ========================================
import numpy as np
if not hasattr(np, '_core'):
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    np._core = numpy.core

import dill
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 현재 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

app = FastAPI(
    title="LinkedIn Fraud Detector API",
    version="3.1.0",
    description="AI-powered job posting fraud detection (Optimized for Railway)"
)

# ========================================
# CORS 설정
# ========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ========================================
# 전역 변수
# ========================================
extractor = None
tfidf = None
embedder = None
model = None
ensemble_models = None
use_ensemble = False
threshold = 0.5
feature_names = []
model_loaded = False
model_version = "unknown"

# ========================================
# 모델 로드 함수
# ========================================
def load_model():
    """모델 로드 (v17 호환)"""
    global extractor, tfidf, embedder, model, ensemble_models
    global use_ensemble, threshold, feature_names, model_loaded, model_version

    logger.info("🔄 모델 로딩 시작...")

    try:
        # 🔥 모델 파일 자동 탐색 (v17 우선)
        possible_files = [
            'model_v17_lightweight.pkl',
            'production_model_v17.pkl',
            'model_v17.pkl',
            'production_model_v13_enhanced_rules.pkl',
            'fraud_detection_render_v2.pkl',
            'model.pkl'
        ]

        pkl_path = None
        for filename in possible_files:
            test_path = os.path.join(current_dir, filename)
            if os.path.exists(test_path):
                pkl_path = test_path
                logger.info(f"✅ 모델 발견: {filename}")
                break

        if pkl_path is None:
            logger.error(f"❌ 모델 파일 없음. 확인 필요: {possible_files}")
            return False

        logger.info(f"📂 로딩 중: {pkl_path}")

        # 모델 로드
        with open(pkl_path, 'rb') as f:
            artifacts = dill.load(f)

        # 버전 확인
        model_version = artifacts.get('version', 'unknown')
        logger.info(f"  ✓ 모델 버전: {model_version}")

        # Feature Extractor (여러 키 시도)
        for key in ['feature_extractor', 'extractor']:
            if key in artifacts:
                extractor = artifacts[key]
                logger.info(f"  ✓ Feature Extractor: {type(extractor).__name__}")
                break

        if extractor is None:
            logger.error("  ❌ Feature Extractor 없음")
            return False

        # TF-IDF (선택적)
        if 'tfidf' in artifacts:
            tfidf = artifacts['tfidf']
            logger.info("  ✓ TF-IDF")

        # FastText Embedder (v17 전용)
        if 'embedder' in artifacts:
            embedder = artifacts['embedder']
            logger.info("  ✓ FastText Embedder")

        # Feature names
        if 'feature_names' in artifacts:
            feature_names = artifacts['feature_names']
            logger.info(f"  ✓ Features: {len(feature_names)}개")

        # Threshold
        if 'threshold' in artifacts:
            threshold = artifacts['threshold']
            logger.info(f"  ✓ Threshold: {threshold:.3f}")

        # Model (Single or Ensemble)
        if 'use_ensemble' in artifacts and artifacts['use_ensemble']:
            use_ensemble = True
            ensemble_models = artifacts.get('ensemble_models', {})
            logger.info(f"  ✓ Ensemble 모드: {len(ensemble_models)}개 모델")
        else:
            model = artifacts.get('model')
            model_name = artifacts.get('model_name', 'Unknown')
            logger.info(f"  ✓ Single 모델: {model_name}")

        if model is None and (not use_ensemble or not ensemble_models):
            logger.error("  ❌ 모델 없음")
            return False

        model_loaded = True
        logger.info("✅ 모델 로드 완료!")
        return True

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False
        return False

# 서버 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    logger.info("🚀 서버 시작...")
    success = load_model()
    if not success:
        logger.warning("⚠️  모델 없이 서버 시작 (헬스체크만 가능)")
    else:
        logger.info("✅ API 준비 완료!")

# ========================================
# Pydantic 모델
# ========================================
class JobPosting(BaseModel):
    title: str = ""
    description: str = ""
    company_profile: str = ""
    salary_range: str = ""
    requirements: str = ""
    benefits: str = ""
    has_company_logo: int = 0
    telecommuting: int = 0
    has_questions: int = 0
    industry: str = ""
    function: str = ""
    location: str = ""
    department: str = ""
    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""

# ========================================
# API 엔드포인트
# ========================================
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🔍 LinkedIn Fraud Detector API",
        "version": "3.1.0",
        "status": "online" if model_loaded else "model not loaded",
        "model": {
            "loaded": model_loaded,
            "version": model_version,
            "extractor": type(extractor).__name__ if extractor else None,
            "features": len(feature_names) if feature_names else 0,
            "mode": "ensemble" if use_ensemble else "single",
            "threshold": float(threshold),
            "has_tfidf": tfidf is not None,
            "has_embedder": embedder is not None
        },
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "reload": "POST /reload"
        }
    }

@app.head("/")
async def head_root():
    return JSONResponse(content={}, status_code=200)

@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "model_version": model_version,
        "extractor": type(extractor).__name__ if extractor else None,
        "tfidf_loaded": tfidf is not None,
        "embedder_loaded": embedder is not None,
        "mode": "ensemble" if use_ensemble else "single",
        "features": len(feature_names) if feature_names else 0,
    }

@app.head("/health")
async def head_health():
    return JSONResponse(content={}, status_code=200)

@app.post("/reload")
async def reload_model():
    """모델 재로드 (관리자용)"""
    logger.info("🔄 모델 재로드 요청...")
    success = load_model()
    return {
        "status": "success" if success else "failed",
        "model_loaded": model_loaded,
        "model_version": model_version
    }

@app.options("/analyze")
async def options_analyze():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.head("/analyze")
async def head_analyze():
    return JSONResponse(content={}, status_code=200)

@app.post("/analyze")
async def analyze_job(job: JobPosting):
    """채용공고 사기 탐지 분석"""
    logger.info(f"📨 분석 요청: {job.title[:50] if job.title else 'No title'}...")

    if not model_loaded or extractor is None:
        logger.error("❌ 모델 미로드")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )

    try:
        # 입력 검증
        if not job.title and not job.description:
            raise HTTPException(
                status_code=400,
                detail="Title or description required"
            )

        # DataFrame 생성
        df = pd.DataFrame([job.dict()])
        logger.info(f"  ✓ Input validated")

        # ========================================
        # Feature 추출 (v17 호환)
        # ========================================

        # 1. Domain Features
        if hasattr(extractor, 'transform'):
            X_domain = extractor.transform(df)
        elif hasattr(extractor, 'extract_all_features'):
            features_dict = extractor.extract_all_features(df.iloc[0].to_dict())
            X_domain = pd.DataFrame([features_dict])
        else:
            raise HTTPException(status_code=500, detail="Incompatible feature extractor")

        logger.info(f"  ✓ Domain features: {X_domain.shape}")

        # 2. TF-IDF (선택적)
        X_tfidf_df = pd.DataFrame()
        if tfidf is not None:
            try:
                texts = (df['title'].fillna('') + ' ' +
                        df['description'].fillna('') + ' ' +
                        df['requirements'].fillna('')).tolist()
                X_tfidf = tfidf.transform(texts)
                X_tfidf_df = pd.DataFrame(
                    X_tfidf.toarray(),
                    columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])],
                    index=X_domain.index
                )
                logger.info(f"  ✓ TF-IDF: {X_tfidf_df.shape}")
            except Exception as e:
                logger.warning(f"  ⚠️  TF-IDF 스킵: {e}")

        # 3. FastText Embedder (v17 전용)
        X_embedder_df = pd.DataFrame()
        if embedder is not None:
            try:
                texts = (df['title'].fillna('') + ' ' +
                        df['description'].fillna('')).tolist()

                if hasattr(embedder, 'transform'):
                    embeddings = embedder.transform(texts)
                elif hasattr(embedder, 'get_embedding'):
                    embeddings = np.array([embedder.get_embedding(t) for t in texts])
                else:
                    embeddings = None

                if embeddings is not None:
                    X_embedder_df = pd.DataFrame(
                        embeddings,
                        columns=[f'embed_{i}' for i in range(embeddings.shape[1])],
                        index=X_domain.index
                    )
                    logger.info(f"  ✓ Embedder: {X_embedder_df.shape}")
            except Exception as e:
                logger.warning(f"  ⚠️  Embedder 스킵: {e}")

        # ========================================
        # 특성 결합
        # ========================================
        dfs_to_concat = [X_domain]
        if not X_tfidf_df.empty:
            dfs_to_concat.append(X_tfidf_df)
        if not X_embedder_df.empty:
            dfs_to_concat.append(X_embedder_df)

        X_combined = pd.concat(dfs_to_concat, axis=1)
        logger.info(f"  ✓ Combined features: {X_combined.shape}")

        # ========================================
        # 예측
        # ========================================
        if use_ensemble and ensemble_models:
            # Ensemble 모드
            probas = []
            for name, m in ensemble_models.items():
                try:
                    proba = m.predict_proba(X_combined)[0, 1]
                    probas.append(proba)
                    logger.info(f"    • {name}: {proba:.4f}")
                except Exception as e:
                    logger.warning(f"    ⚠️  {name} 실패: {e}")

            if not probas:
                raise HTTPException(status_code=500, detail="All ensemble models failed")

            balanced_proba = float(np.mean(probas))
            logger.info(f"  ✓ Ensemble 평균: {balanced_proba:.4f}")
        else:
            # Single 모드
            if model is None:
                raise HTTPException(status_code=503, detail="Model not available")

            balanced_proba = float(model.predict_proba(X_combined)[0, 1])
            logger.info(f"  ✓ Probability: {balanced_proba:.4f}")

        # ========================================
        # 판정
        # ========================================
        prediction = 1 if balanced_proba >= threshold else 0

        if prediction == 1:
            if balanced_proba > 0.80:
                action = 'BLOCK'
                reason = 'Very high fraud probability - Immediate block recommended'
                risk_level = 'CRITICAL'
            else:
                action = 'REVIEW'
                reason = 'High risk - Manual review strongly recommended'
                risk_level = 'HIGH'
        else:
            if balanced_proba > 0.30:
                action = 'REVIEW'
                reason = 'Medium risk - Consider manual review'
                risk_level = 'MEDIUM'
            else:
                action = 'PASS'
                reason = 'Appears to be a legitimate job posting'
                risk_level = 'LOW'

        result = {
            'action': action,
            'reason': reason,
            'risk_level': risk_level,
            'probability': balanced_proba,
            'prediction': prediction,
            'threshold': float(threshold),
            'model_version': model_version
        }

        logger.info(f"  ✅ 결과: {action} (prob={balanced_proba:.3f}, risk={risk_level})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ========================================
# Catch-all handlers
# ========================================
@app.options("/{path:path}")
async def catch_all_options(path: str):
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.head("/{path:path}")
async def catch_all_head(path: str):
    return JSONResponse(content={}, status_code=200)

# ========================================
# 미들웨어
# ========================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📥 {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"📤 {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 서버 시작: 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)