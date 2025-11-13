"""
Model Loader for Hugging Face Hub
"""

from huggingface_hub import hf_hub_download
import pickle
import logging
import os
import shutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Model configuration
REPO_ID = "functionss/fake-job-detector"
FILENAME = "model_v17_final.pkl"

def download_model(force=False):
    """
    Download the trained model from Hugging Face Hub

    Args:
        force (bool): Force re-download even if file exists

    Returns:
        str: Path to the downloaded model
    """

    if os.path.exists(FILENAME) and not force:
        logger.info(f"✅ Model already exists: {FILENAME}")
        return FILENAME

    logger.info("📥 Downloading model from Hugging Face Hub...")
    logger.info(f"   Repository: {REPO_ID}")
    logger.info(f"   File: {FILENAME}")
    logger.info(f"   Size: ~829 MB")
    logger.info(f"   This may take 2-5 minutes...")

    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir="./cache"
        )

        # Copy to current directory for easier access
        shutil.copy(model_path, FILENAME)

        logger.info(f"✅ Download complete: {FILENAME}")
        return FILENAME

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error(f"💡 Alternative: Download manually from:")
        logger.error(f"   https://huggingface.co/{REPO_ID}")
        raise


def load_model() -> Optional[Dict[str, Any]]:
    """
    Load model from Hugging Face Hub or local cache

    Returns:
        Dict containing model artifacts or None if failed
    """
    try:
        logger.info(f"🔍 모델 로딩 중...")

        # 1. Download model if not exists
        model_path = download_model()

        logger.info(f"📦 모델 파일 언팩 중...")

        # 2. Load the pickle file
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)

        logger.info(f"✅ 모델 로드 성공!")
        logger.info(f"   Version: {model_artifacts.get('version', 'unknown')}")
        logger.info(f"   Threshold: {model_artifacts.get('threshold', 'unknown')}")
        logger.info(f"   Components: {', '.join(model_artifacts.keys())}")

        return model_artifacts

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        logger.error(f"   Repository: https://huggingface.co/{REPO_ID}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # 직접 실행 시 다운로드만 수행
    download_model()