"""
Fake Job Detector - Model Downloader
Downloads the trained model from Hugging Face Hub
"""

from huggingface_hub import hf_hub_download
import os

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
        print(f"✅ Model already exists: {FILENAME}")
        print(f"   Use force=True to re-download")
        return FILENAME

    print("📥 Downloading model from Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    print(f"   File: {FILENAME}")
    print(f"   Size: ~829 MB")
    print(f"   This may take 2-5 minutes...")

    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir="./cache"
        )

        # Copy to current directory for easier access
        import shutil
        shutil.copy(model_path, FILENAME)

        print(f"\n✅ Download complete!")
        print(f"   Location: {FILENAME}")
        return FILENAME

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\n💡 Alternative: Download manually from:")
        print(f"   https://huggingface.co/{REPO_ID}")
        raise

if __name__ == "__main__":
    download_model()
