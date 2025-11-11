"""
production_model_v13.pkl을 Render용 dill로 변환 (Windows 호환 + BERT 제거)
"""
import pickle
import dill
import sys
import io
import torch
from pathlib import Path, PureWindowsPath, PurePosixPath

# Windows에서 PosixPath 호환
class PosixPath(PurePosixPath):
    def __new__(cls, *args):
        return PureWindowsPath(*args)

import pathlib
pathlib.PosixPath = PosixPath

# models.py import
import models
sys.modules['models'] = models
from models import (FeatureExtractor, BERTEmbedder, FocalLossClassifier,
                    AdvancedFeatureExtractor, ProductionMonitor)

print("🔄 production_model_v13.pkl 로딩...")

class WindowsCPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            return PosixPath
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

try:
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'map_location': 'cpu'})

    with open('production_model_v13.pkl', 'rb') as f:
        artifacts = WindowsCPU_Unpickler(f).load()

    torch.load = original_load
    print("✅ 로드 완료!")

except Exception as e:
    print(f"❌ 최종 실패: {e}")
    import traceback
    traceback.print_exc()
    exit()

# FeatureExtractor 재생성
print("\n🔧 FeatureExtractor 재생성...")
old_extractor = artifacts['feature_extractor']

try:
    keywords = getattr(old_extractor, 'keywords', [])
    ind_risk = getattr(old_extractor, 'ind_risk', {})
    func_risk = getattr(old_extractor, 'func_risk', {})
    overall_rate = getattr(old_extractor, 'overall_rate', 0.1)
    thresholds = getattr(old_extractor, 'thresholds', {
        'caps': 0.3, 'exclaim': 3, 'polarity': 0.5, 'subjectivity': 0.6
    })
    print(f"  ✓ Keywords: {len(keywords)}개")
except:
    keywords = []
    ind_risk = {}
    func_risk = {}
    overall_rate = 0.1
    thresholds = {'caps': 0.3, 'exclaim': 3, 'polarity': 0.5, 'subjectivity': 0.6}

new_extractor = FeatureExtractor(
    keywords=keywords,
    ind_risk=ind_risk,
    func_risk=func_risk,
    overall_rate=overall_rate,
    thresholds=thresholds
)
print("  ✓ 생성 완료")

# BERTEmbedder (모델 제거 버전)
print("\n🤖 BERTEmbedder 설정...")
bert_embedder = BERTEmbedder(n_components=64)

if 'pca' in artifacts:
    bert_embedder.pca = artifacts['pca']
    bert_embedder.pca_fitted = True
    print("  ✓ PCA 복사 완료")

# ⚠️ CRITICAL: BERT 모델 제거 (pickle 불가능)
bert_embedder.model = None
bert_embedder.model_name = 'all-MiniLM-L6-v2'  # Render에서 재로드용
print("  ✓ BERT 모델 제거 (Render에서 재초기화)")

# 모델 가져오기
print("\n📦 모델 구성...")
try:
    if artifacts.get('use_ensemble'):
        ensemble_models = artifacts['ensemble_models']
        models_list = list(ensemble_models.values())
        xgb_model = models_list[0] if len(models_list) > 0 else artifacts['model']
        lgbm_model = models_list[1] if len(models_list) > 1 else None
        cat_model = models_list[2] if len(models_list) > 2 else None
        print(f"  ✓ Ensemble: {len(models_list)}개 모델")
    else:
        xgb_model = artifacts['model']
        lgbm_model = None
        cat_model = None
        print(f"  ✓ Single 모델")
except:
    xgb_model = artifacts.get('model')
    lgbm_model = None
    cat_model = None

# Render용 artifacts 구성
print("\n📦 Render용 패키징...")
render_artifacts = {
    'domain_extractor': new_extractor,
    'bert_embedder': bert_embedder,
    'selector': artifacts.get('selector', None),
    'models_balanced': {
        'xgb': xgb_model,
        'lgbm': lgbm_model,
        'cat': cat_model,
        'nn': None,
        'weights': {
            'xgb': 0.4 if lgbm_model else 1.0,
            'lgbm': 0.3 if lgbm_model else 0.0,
            'cat': 0.2 if cat_model else 0.0,
            'nn': 0.1 if lgbm_model else 0.0
        }
    }
}

# dill로 저장
print("\n💾 dill로 저장 중...")
try:
    with open('fraud_detection_render_v2.pkl', 'wb') as f:
        dill.dump(render_artifacts, f, recurse=True)

    import os
    file_size = os.path.getsize('fraud_detection_render_v2.pkl')
    print(f"✅ 완료: fraud_detection_render_v2.pkl ({file_size / 1024 / 1024:.1f} MB)")
    print("\n🚀 Git 명령어:")
    print("  git add fraud_detection_render_v2.pkl main.py models.py requirements.txt")
    print("  git commit -m 'Add dill model'")
    print("  git push")

except Exception as e:
    print(f"❌ 저장 실패: {e}")
    import traceback
    traceback.print_exc()