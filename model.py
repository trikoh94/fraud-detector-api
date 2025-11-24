"""
모델 로더 - v33 (Local File)
GitHub repo에 있는 model_v33.pkl 직접 로드
"""

import joblib
import logging
import re
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# V29FeatureExtractor 클래스 (필수!)
# ============================================================================

class V29FeatureExtractor:
    """v24 + Strong Signal Interactions"""

    def __init__(self):
        self.fraud_patterns = {
            'payment_scam': [
                r'pay.*upfront', r'send.*money.*first', r'deposit.*before',
                r'registration.*fee', r'training.*kit.*\$', r'starter.*package',
                r'certification.*fee', r'background.*check.*fee'
            ],
            'personal_info': [
                r'ssn|social\s*security', r'bank.*account.*number',
                r'routing.*number', r'credit.*card.*verification'
            ],
            'wfh_scam': [
                r'work.*home.*easy.*money', r'\$\d+.*week.*guaranteed',
                r'no.*experience.*high.*pay', r'earn.*while.*you.*sleep'
            ],
            'urgency': [
                r'act.*now.*limited', r'offer.*expires.*today',
                r'immediate.*start.*required'
            ],
            'mlm': [
                r'recruit.*earn', r'build.*team', r'downline',
                r'unlimited.*income.*potential'
            ],
            'too_good': [
                r'\$\d{3,4}\+?\s*(per|/)?\s*(day|daily)',
                r'earn.*\$\d+.*hour.*no.*experience'
            ],
            'external_contact': [
                r'whatsapp.*\+?\d+', r'telegram.*@\w+',
                r'text.*\d{3}[-.]?\d{3}'
            ]
        }

        self.legitimate_signals = {
            'benefits': ['benefits', '401k', 'insurance', 'pto', 'vacation', 'medical', 'dental'],
            'requirements': ['bachelor', 'master', 'years experience', 'certification', 'degree'],
            'company': ['founded', 'headquarters', 'employees', 'industry leader', 'established']
        }

        self.suspicious = {
            'free_email': r'@(gmail|yahoo|hotmail|outlook|aol)\.com',
            'excessive_salary': r'\$(\d{3,4})\s*(per|/)?\s*(day|hour)',
            'phone_in_desc': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'all_caps': r'\b[A-Z]{5,}\b',
            'many_exclaim': r'[!?]{2,}'
        }

    def extract_basic_stats(self, text):
        if not text:
            return {f'stat_{i}': 0 for i in range(10)}

        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        return {
            'stat_word_count': len(words),
            'stat_avg_word_len': sum(len(w) for w in words) / max(len(words), 1),
            'stat_sentence_count': len(sentences),
            'stat_avg_sent_len': len(words) / max(len(sentences), 1),
            'stat_unique_ratio': len(set(words)) / max(len(words), 1),
            'stat_upper_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'stat_digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'stat_punct_ratio': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
            'stat_space_errors': text.count('  ') + len(re.findall(r'\s[.,!?]', text)),
            'stat_question_count': text.count('?')
        }

    def extract_fraud_signals(self, text, prefix=''):
        if not text:
            return {}

        text_lower = text.lower()
        features = {}

        for category, patterns in self.fraud_patterns.items():
            count = sum(1 for p in patterns if re.search(p, text_lower))
            features[f'{prefix}fraud_{category}'] = count

        for category, keywords in self.legitimate_signals.items():
            count = sum(text_lower.count(kw) for kw in keywords)
            features[f'{prefix}legit_{category}'] = count

        for name, pattern in self.suspicious.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            features[f'{prefix}sus_{name}'] = count

        return features

    def extract_contextual(self, job_data):
        features = {}

        title = str(job_data.get('title', ''))
        desc = str(job_data.get('description', ''))
        req = str(job_data.get('requirements', ''))
        company = str(job_data.get('company_profile', ''))
        benefits = str(job_data.get('benefits', ''))

        features['ctx_has_title'] = int(len(title) > 5)
        features['ctx_has_desc'] = int(len(desc) > 100)
        features['ctx_has_req'] = int(len(req) > 50)
        features['ctx_has_benefits'] = int(len(benefits) > 20)
        features['ctx_has_company'] = int(len(company) > 50)
        features['ctx_completeness'] = sum([
            features['ctx_has_title'], features['ctx_has_desc'],
            features['ctx_has_req'], features['ctx_has_benefits'],
            features['ctx_has_company']
        ])

        total_len = len(desc) + len(req) + len(benefits)
        if total_len > 0:
            features['ctx_desc_ratio'] = len(desc) / total_len
            features['ctx_req_ratio'] = len(req) / total_len
        else:
            features['ctx_desc_ratio'] = 0
            features['ctx_req_ratio'] = 0

        salary_match = re.search(r'\$(\d+)', desc)
        if salary_match:
            amount = int(salary_match.group(1))
            features['ctx_has_salary'] = 1
            if 'day' in desc.lower() or 'daily' in desc.lower():
                features['ctx_daily_pay'] = 1
                features['ctx_hourly_est'] = amount / 8
            elif 'hour' in desc.lower():
                features['ctx_hourly_est'] = amount
            else:
                features['ctx_daily_pay'] = 0
                features['ctx_hourly_est'] = 0
        else:
            features['ctx_has_salary'] = 0
            features['ctx_daily_pay'] = 0
            features['ctx_hourly_est'] = 0

        features['ctx_email_count'] = len(re.findall(r'\b[\w\.-]+@[\w\.-]+\b', desc))
        features['ctx_phone_count'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', desc))

        return features

    def extract_critical_combos(self, job_data, desc_fraud, title_fraud):
        has_logo = int(job_data.get('has_company_logo', 0))
        company = str(job_data.get('company_profile', ''))
        req = str(job_data.get('requirements', ''))

        fraud_total = sum(v for v in desc_fraud.values() if isinstance(v, int))
        legit_total = sum(v for k, v in desc_fraud.items() if 'legit' in k)

        return {
            'combo_fraud_total': fraud_total,
            'combo_legit_total': legit_total,
            'combo_fraud_to_legit': fraud_total / max(legit_total, 1),
            'combo_payment_no_company': int(
                desc_fraud.get('desc_fraud_payment_scam', 0) > 0 and len(company) < 20
            ),
            'combo_urgent_no_logo': int(
                desc_fraud.get('desc_fraud_urgency', 0) > 0 and has_logo == 0
            ),
            'combo_high_pay_no_req': int(
                desc_fraud.get('desc_sus_excessive_salary', 0) > 0 and len(req) < 50
            ),
            'combo_external_no_logo': int(
                desc_fraud.get('desc_fraud_external_contact', 0) > 0 and has_logo == 0
            )
        }

    def extract_company_quality(self, job_data):
        company = str(job_data.get('company_profile', '')).lower()
        features = {}

        features['v24_company_has_year'] = int(bool(
            re.search(r'19\d{2}|20\d{2}', company)
        ))

        features['v24_company_has_employees'] = int(bool(
            re.search(r'\d+\s*(employee|staff|people|team\s+member)', company)
        ))

        features['v24_company_has_location'] = int(bool(
            re.search(r'(headquarter|office|located\s+in|based\s+in)', company)
        ))

        features['v24_company_has_metrics'] = int(bool(
            re.search(r'(revenue|profit|market|customer|client)s?\s*\d+', company)
        ))

        words = company.split()
        sentences = [s for s in re.split(r'[.!?]+', company) if s.strip()]

        if sentences and words:
            features['v24_company_info_density'] = len(words) / len(sentences)
        else:
            features['v24_company_info_density'] = 0

        return features

    def extract_no_logo_signals(self, job_data):
        has_logo = int(job_data.get('has_company_logo', 0))
        features = {}

        if has_logo == 0:
            company = str(job_data.get('company_profile', ''))
            req = str(job_data.get('requirements', ''))

            features['v24_no_logo_no_company'] = int(len(company) < 100)
            features['v24_no_logo_no_req'] = int(len(req) < 50)
        else:
            features['v24_no_logo_no_company'] = 0
            features['v24_no_logo_no_req'] = 0

        return features

    def extract_strong_signals_v29(self, job_data, company_quality):
        has_logo = int(job_data.get('has_company_logo', 0))
        has_year = company_quality['v24_company_has_year']
        company = str(job_data.get('company_profile', ''))
        benefits = str(job_data.get('benefits', ''))
        req = str(job_data.get('requirements', ''))

        features = {}

        features['v29_very_legit'] = int(
            has_logo == 1 and has_year == 1 and len(company) > 200
        )

        features['v29_complete_posting'] = int(
            len(benefits) > 50 and len(req) > 100
        )

        features['v29_very_suspicious'] = int(
            has_logo == 0 and len(company) < 50 and len(req) < 50
        )

        return features

    def extract_all_features(self, job_data):
        """83개 features"""
        desc = str(job_data.get('description', ''))
        title = str(job_data.get('title', ''))

        features = {}
        features.update(self.extract_basic_stats(desc))

        desc_fraud = self.extract_fraud_signals(desc, 'desc_')
        title_fraud = self.extract_fraud_signals(title, 'title_')
        features.update(desc_fraud)
        features.update(title_fraud)

        features.update(self.extract_contextual(job_data))

        features['meta_has_logo'] = int(job_data.get('has_company_logo', 0))
        features['meta_telecommuting'] = int(job_data.get('telecommuting', 0))
        features['meta_has_questions'] = int(job_data.get('has_questions', 0))

        combos = self.extract_critical_combos(job_data, desc_fraud, title_fraud)
        features.update(combos)

        company_quality = self.extract_company_quality(job_data)
        features.update(company_quality)

        features.update(self.extract_no_logo_signals(job_data))
        features.update(self.extract_strong_signals_v29(job_data, company_quality))

        return features


# ============================================================================
# 모델 로드
# ============================================================================

def load_model():
    """로컬 파일에서 v33 모델 로드"""

    try:
        # ✅ 간단! GitHub repo에 있는 파일 직접 로드
        model_path = Path(__file__).parent / 'model_v33.pkl'

        logger.info(f"📦 모델 로딩 중: {model_path}")

        if not model_path.exists():
            logger.error(f"❌ 파일 없음: {model_path}")
            return None

        # 파일 크기 확인
        file_size = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"📦 파일 크기: {file_size:.1f}MB")

        # Joblib로 로드
        artifacts = joblib.load(model_path)

        # ✅ Feature extractor는 직접 생성 (pickle 문제 회피)
        artifacts['feature_extractor'] = V29FeatureExtractor()

        logger.info("=" * 70)
        logger.info("✅ 모델 로드 성공!")
        logger.info(f"📊 Version: {artifacts.get('version', 'v33')}")
        logger.info(f"🎯 Threshold: {artifacts.get('threshold', 0.045):.4f}")

        metrics = artifacts.get('metrics', {})
        if metrics:
            logger.info(f"🏆 F1 Score: {metrics.get('f1', 0):.2%}")
            logger.info(f"🔥 Recall: {metrics.get('recall', 0):.2%}")
            logger.info(f"⚡ Precision: {metrics.get('precision', 0):.2%}")

        logger.info(f"📊 Domain Features: {len(artifacts['feature_columns'])}")
        logger.info(f"📊 TF-IDF: {artifacts['tfidf'].max_features}")
        logger.info(f"📊 FastText: {artifacts['fasttext_embedder'].vector_size} (vocab: {artifacts['fasttext_embedder'].vocab_size:,})")
        logger.info(f"📊 Total: {len(artifacts['feature_columns']) + artifacts['tfidf'].max_features + artifacts['fasttext_embedder'].vector_size}")

        return artifacts

    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None