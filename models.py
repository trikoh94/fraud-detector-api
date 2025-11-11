"""
모델 클래스 정의 파일 (Enhanced - 학습 파일과 호환)
"""
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from functools import lru_cache


@lru_cache(maxsize=1000)
def get_sentiment(text):
    """캐싱된 감성 분석"""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


# ============================================================================
# 🆕 AdvancedFeatureExtractor (학습 파일과 동일)
# ============================================================================

class AdvancedFeatureExtractor:
    """🔥 Enhanced: 논문 수준 Rule-based Features (40+)"""

    def __init__(self):
        # ============================================================
        # 1. 🚨 CRITICAL SCAM PATTERNS (논문 기반)
        # ============================================================
        self.critical_patterns = {
            # Payment/Money requests
            'payment_request': [
                r'upfront\s*(fee|payment|cost|charge)',
                r'send\s*(money|payment|cash|funds)',
                r'wire\s*transfer',
                r'registration\s*(fee|cost)',
                r'processing\s*fee',
                r'training\s*fee',
                r'background\s*check\s*fee',
                r'pay.*before.*start',
                r'deposit.*required',
                r'advance.*payment',
                r'starter\s*kit.*\$',
                r'investment.*required',
            ],

            # Personal Info requests (매우 위험)
            'personal_info': [
                r'ssn|social\s*security\s*number',
                r'bank\s*account\s*(number|details)',
                r'credit\s*card\s*(number|info)',
                r'passport\s*(number|copy)',
                r'driver.*license\s*number',
                r'tax\s*id',
                r'routing\s*number',
                r'pin\s*number',
                r'date\s*of\s*birth.*\d{2}',
            ],

            # Work-from-home scams
            'wfh_scam': [
                r'work\s*from\s*home.*easy\s*money',
                r'make.*\$\d+.*per\s*(day|week|hour).*home',
                r'no\s*experience.*high\s*pay',
                r'earn.*unlimited',
                r'be\s*your\s*own\s*boss',
                r'financial\s*freedom',
                r'passive\s*income',
                r'residual\s*income',
            ],

            # Package/Reshipping scams
            'reshipping': [
                r'package.*handler',
                r'shipping.*coordinator',
                r'repack.*items',
                r'receive.*packages.*home',
                r'quality.*control.*packages',
                r'warehouse.*home',
            ],

            # Money transfer/mule
            'money_mule': [
                r'transfer.*funds',
                r'process.*payments',
                r'payment.*processor',
                r'financial.*agent',
                r'receive.*payments.*forward',
                r'cash.*handling',
            ],
        }

        # ============================================================
        # 2. 🔍 SUSPICIOUS KEYWORDS (카테고리별)
        # ============================================================
        self.fraud_keywords = {
            # 긴급성/압박
            'urgency': [
                'urgent', 'immediately', 'asap', 'hurry', 'act now',
                'limited time', 'limited spots', 'first come',
                'don\'t wait', 'apply now', 'instant', 'right away'
            ],

            # 돈 관련 (의심)
            'payment': [
                'pay upfront', 'send payment', 'deposit required',
                'processing fee', 'registration fee', 'training fee',
                'starter kit', 'pay before', 'advance payment'
            ],

            # 너무 좋은 조건
            'too_good': [
                'guaranteed income', 'easy money', 'no experience needed',
                'unlimited earning', 'high income', 'fast cash',
                'make money fast', 'get rich', 'financial freedom',
                'no skills required', 'anyone can do', 'simple work'
            ],

            # 모호한 설명
            'vague': [
                'various duties', 'multiple tasks', 'and more',
                'other duties', 'flexible role', 'diverse responsibilities'
            ],

            # 외부 커뮤니케이션 요구 (suspicious)
            'external_comm': [
                'whatsapp', 'telegram', 'text me', 'call me at',
                'email me at.*@gmail', 'contact via', 'reach me at'
            ],

            # Fake urgency
            'fake_urgency': [
                'hiring immediately', 'start immediately', 'begin right away',
                'same day start', 'instant hire'
            ],
        }

        # ============================================================
        # 3. 📧 CONTACT PATTERN ANALYSIS
        # ============================================================
        self.suspicious_patterns = {
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'money': r'\$\d+',
            'free_email': r'@(gmail|yahoo|hotmail|outlook|aol|icloud)\.com',
            'suspicious_domain': r'@(mail\.com|inbox\.com|protonmail)',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+',
            'short_url': r'bit\.ly|tinyurl|goo\.gl',
        }

        # ============================================================
        # 4. 💰 SALARY RED FLAGS
        # ============================================================
        self.salary_patterns = {
            'unrealistic_high': r'\$[5-9]\d{2,}.*(?:per\s*hour|/hr)',  # $500+/hr
            'unrealistic_week': r'\$[2-9]\d{3,}.*(?:per\s*week|/week)',  # $2000+/week
            'commission_only': r'commission\s*only|100%\s*commission',
            'vague_salary': r'competitive|negotiable|depends|varies',
        }

        # ============================================================
        # 5. 🏢 COMPANY INFORMATION RED FLAGS
        # ============================================================
        self.company_flags = {
            'generic_names': [
                'hiring now', 'jobs available', 'opportunity',
                'work from home', 'remote jobs', 'hiring team'
            ],
            'no_website_indicators': [
                'no website', 'coming soon', 'under construction'
            ],
        }

        # ============================================================
        # 6. 📝 LINGUISTIC FEATURES (품질 체크)
        # ============================================================
        self.quality_patterns = {
            'excessive_caps': r'[A-Z]{5,}',  # 5+ 연속 대문자
            'excessive_exclamation': r'!{2,}',  # !! 이상
            'excessive_punctuation': r'[.!?]{3,}',  # ...!!!
            'broken_english': [
                r'\b(?:much|very)\s+(?:much|very)\b',  # very very
                r'\bno\s+have\b',
                r'\bmake\s+to\b',
            ],
        }

    @lru_cache(maxsize=1000)
    def extract_text_features(self, text):
        if not isinstance(text, str) or not text.strip():
            return {
                'word_count': 0, 'char_count': 0, 'avg_word_length': 0,
                'uppercase_ratio': 0, 'digit_ratio': 0,
                'sentence_count': 0, 'exclamation_count': 0,
                'sentiment_polarity': 0, 'sentiment_subjectivity': 0
            }

        words = text.split()
        word_count = len(words)
        char_count = len(text)

        try:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
        except:
            sentiment_polarity = 0
            sentiment_subjectivity = 0

        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / char_count if char_count > 0 else 0,
            'sentence_count': max(text.count('.') + text.count('!') + text.count('?'), 1),
            'exclamation_count': text.count('!'),
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity
        }

    def check_critical_patterns(self, text):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        critical_count = 0
        for category, patterns in self.critical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    critical_count += 1
        return critical_count

    def extract_fraud_signals(self, text):
        """🆕 강화된 사기 시그널 추출"""
        if not isinstance(text, str):
            text = ''

        text_lower = text.lower()
        fraud_signals = {}

        # 1. 키워드 기반
        for category, keywords in self.fraud_keywords.items():
            count = sum(text_lower.count(kw.lower()) for kw in keywords)
            fraud_signals[f'{category}_keywords'] = count

        # 2. 패턴 기반
        for pattern_name, pattern in self.suspicious_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            fraud_signals[f'{pattern_name}_count'] = matches

        # 3. Critical patterns (가중치 높음)
        fraud_signals['critical_pattern_count'] = self.check_critical_patterns(text)

        # 4. 🆕 Salary red flags
        fraud_signals['salary_red_flags'] = self._check_salary_flags(text)

        # 5. 🆕 Quality issues
        fraud_signals['quality_issues'] = self._check_quality_issues(text)

        # 6. 🆕 External communication requests
        fraud_signals['external_comm_requests'] = self._check_external_comm(text)

        return fraud_signals

    def _check_salary_flags(self, text):
        """급여 관련 red flag 체크"""
        count = 0
        for pattern_name, pattern in self.salary_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    def _check_quality_issues(self, text):
        """텍스트 품질 이슈 체크"""
        count = 0

        # Excessive caps/punctuation
        for pattern_name, pattern in self.quality_patterns.items():
            if pattern_name in ['excessive_caps', 'excessive_exclamation', 'excessive_punctuation']:
                matches = len(re.findall(pattern, text))
                count += min(matches, 3)  # Cap at 3

        # Broken English patterns
        for pattern in self.quality_patterns.get('broken_english', []):
            if re.search(pattern, text, re.IGNORECASE):
                count += 1

        return count

    def _check_external_comm(self, text):
        """외부 커뮤니케이션 요청 체크"""
        count = 0
        text_lower = text.lower()

        for keyword in self.fraud_keywords.get('external_comm', []):
            if keyword in text_lower:
                count += 1

        return count

    def check_company_flags(self, company_profile, title):
        """회사 정보 red flag"""
        flags = 0

        if not company_profile or len(str(company_profile).strip()) < 20:
            flags += 1  # 회사 설명 너무 짧음

        title_lower = str(title).lower()
        for generic in self.company_flags['generic_names']:
            if generic in title_lower:
                flags += 1

        return flags

    def check_requirements_mismatch(self, description, requirements):
        """요구사항과 설명의 불일치"""
        desc_len = len(str(description))
        req_len = len(str(requirements))

        # 설명은 길지만 요구사항이 거의 없음
        if desc_len > 500 and req_len < 50:
            return 1

        # 요구사항이 설명보다 훨씬 김 (이상함)
        if req_len > desc_len * 2:
            return 1

        return 0

    def extract_all_features(self, job_data):
        """🆕 전체 특성 추출 (40+ features로 확장)"""
        title = str(job_data.get('title', ''))
        description = str(job_data.get('description', ''))
        requirements = str(job_data.get('requirements', ''))
        company_profile = str(job_data.get('company_profile', ''))
        salary_range = str(job_data.get('salary_range', ''))

        # 기존 특성들
        title_features = self.extract_text_features(title)
        desc_features = self.extract_text_features(description)
        req_features = self.extract_text_features(requirements)

        desc_fraud = self.extract_fraud_signals(description)
        title_fraud = self.extract_fraud_signals(title)
        salary_fraud = self.extract_fraud_signals(salary_range)

        all_features = {
            # 기본 텍스트 특성
            'title_word_count': title_features['word_count'],
            'title_char_count': title_features['char_count'],
            'title_uppercase_ratio': title_features['uppercase_ratio'],
            'title_exclamation_count': title_features['exclamation_count'],

            'desc_word_count': desc_features['word_count'],
            'desc_char_count': desc_features['char_count'],
            'desc_avg_word_length': desc_features['avg_word_length'],
            'desc_uppercase_ratio': desc_features['uppercase_ratio'],
            'desc_digit_ratio': desc_features['digit_ratio'],
            'desc_sentence_count': desc_features['sentence_count'],
            'desc_exclamation_count': desc_features['exclamation_count'],
            'desc_sentiment_polarity': desc_features['sentiment_polarity'],
            'desc_sentiment_subjectivity': desc_features['sentiment_subjectivity'],

            'req_word_count': req_features['word_count'],
            'req_char_count': req_features['char_count'],

            # 🆕 강화된 사기 시그널
            'desc_urgency_keywords': desc_fraud['urgency_keywords'],
            'desc_payment_keywords': desc_fraud['payment_keywords'],
            'desc_too_good_keywords': desc_fraud['too_good_keywords'],
            'desc_vague_keywords': desc_fraud.get('vague_keywords', 0),
            'desc_external_comm_keywords': desc_fraud.get('external_comm_keywords', 0),
            'desc_fake_urgency_keywords': desc_fraud.get('fake_urgency_keywords', 0),

            'desc_phone_count': desc_fraud['phone_count'],
            'desc_email_count': desc_fraud['email_count'],
            'desc_money_count': desc_fraud['money_count'],
            'desc_free_email_count': desc_fraud['free_email_count'],
            'desc_suspicious_domain_count': desc_fraud.get('suspicious_domain_count', 0),
            'desc_url_count': desc_fraud.get('url_count', 0),
            'desc_short_url_count': desc_fraud.get('short_url_count', 0),

            'desc_critical_pattern_count': desc_fraud['critical_pattern_count'],
            'title_critical_pattern_count': title_fraud['critical_pattern_count'],

            # 🆕 급여 red flags
            'salary_red_flags': desc_fraud.get('salary_red_flags', 0) + salary_fraud.get('salary_red_flags', 0),

            # 🆕 품질 이슈
            'desc_quality_issues': desc_fraud.get('quality_issues', 0),
            'title_quality_issues': title_fraud.get('quality_issues', 0),

            # 🆕 외부 커뮤니케이션
            'external_comm_requests': desc_fraud.get('external_comm_requests', 0),

            # 🆕 회사 정보 flags
            'company_info_flags': self.check_company_flags(company_profile, title),

            # 🆕 요구사항 불일치
            'requirements_mismatch': self.check_requirements_mismatch(description, requirements),

            # Boolean features
            'has_company_logo': int(job_data.get('has_company_logo', 0)),
            'telecommuting': int(job_data.get('telecommuting', 0)),
            'has_questions': int(job_data.get('has_questions', 0)),

            # Ratios
            'desc_to_req_ratio': desc_features['word_count'] / max(req_features['word_count'], 1),

            # 🆕 총 사기 시그널 (더 많은 요소)
            'total_fraud_signals': (
                    sum(desc_fraud.values()) +
                    sum(title_fraud.values()) +
                    sum(salary_fraud.values())
            ),

            'has_salary_info': int(bool(job_data.get('salary_range'))),

            # 🆕 회사 프로필 길이
            'company_profile_length': len(company_profile),
            'has_company_profile': int(len(company_profile) > 50),
        }

        return all_features

    def transform(self, df):
        """DataFrame 변환 (API 호환)"""
        features_list = []
        for idx, row in df.iterrows():
            features = self.extract_all_features(row.to_dict())
            features_list.append(features)

        return pd.DataFrame(features_list, index=df.index)


# ============================================================================
# Legacy FeatureExtractor (기존 API와의 호환성 유지)
# ============================================================================

class FeatureExtractor:
    """도메인 특성 추출기 (Legacy - 기존 모델용)"""

    def __init__(self, keywords, ind_risk, func_risk, overall_rate, thresholds):
        self.keywords = keywords
        self.ind_risk = ind_risk
        self.func_risk = func_risk
        self.overall_rate = overall_rate
        self.thresholds = thresholds

    def extract_text_features(self, text, prefix=''):
        """텍스트 특성 추출"""
        if pd.isna(text) or text == '':
            return self._empty_features(prefix)

        text_str = str(text)
        text_lower = text_str.lower()
        words = text_str.split()
        word_count = len(words)
        sentence_count = max(len(re.findall(r'[.!?]+', text_str)), 1)
        text_length = len(text_str)

        polarity, subjectivity = get_sentiment(text_str)

        emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))
        phones = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_str))
        urls = len(re.findall(r'http[s]?://[^\s]+', text_str))

        keyword_cnt = sum(kw in text_lower for kw in self.keywords)
        caps_ratio = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)

        urgency_words = ['urgent', 'hurry', 'now', 'asap', 'immediately', 'limited time', 'act now', 'quick']
        urgency_raw = sum(w in text_lower for w in urgency_words)

        pressure_words = ['must', 'required', 'guarantee', 'easy', 'fast', 'quick', 'instant']
        pressure_raw = sum(w in text_lower for w in pressure_words)

        money_words = ['earn', 'income', 'profit', 'cash', 'money', '$', 'dollar', 'paid', 'pay']
        money_raw = sum(w in text_lower for w in money_words)

        exaggeration = ['amazing', 'incredible', 'unbelievable', 'guaranteed', '100%', 'unlimited', 'free',
                        'higher than']
        exag_raw = sum(w in text_lower for w in exaggeration)

        length_penalty = max(1.0, 200.0 / max(text_length, 50))

        urgency_weighted = urgency_raw * length_penalty * 3.0
        pressure_weighted = pressure_raw * length_penalty * 2.0
        money_weighted = money_raw * length_penalty * 2.5
        exag_weighted = exag_raw * length_penalty * 2.5

        combo_score = 0
        if urgency_raw > 0 and money_raw > 0:
            combo_score += 5
        if urgency_raw > 0 and exag_raw > 0:
            combo_score += 4
        if money_raw > 1 and exag_raw > 0:
            combo_score += 3
        if urgency_raw > 1 and money_raw > 1:
            combo_score += 6

        return {
            f'{prefix}length': text_length,
            f'{prefix}word_count': word_count,
            f'{prefix}sentence_count': sentence_count,
            f'{prefix}avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            f'{prefix}avg_sent_len': word_count / sentence_count,
            f'{prefix}caps_ratio': caps_ratio,
            f'{prefix}high_caps': int(caps_ratio > self.thresholds['caps']),
            f'{prefix}exclaim': text_str.count('!'),
            f'{prefix}high_exclaim': int(text_str.count('!') > self.thresholds['exclaim']),
            f'{prefix}question': text_str.count('?'),
            f'{prefix}keyword': keyword_cnt,
            f'{prefix}has_keyword': int(keyword_cnt > 0),
            f'{prefix}urgency_raw': urgency_raw,
            f'{prefix}urgency': urgency_weighted,
            f'{prefix}pressure_raw': pressure_raw,
            f'{prefix}pressure': pressure_weighted,
            f'{prefix}money_raw': money_raw,
            f'{prefix}money': money_weighted,
            f'{prefix}exag_raw': exag_raw,
            f'{prefix}exag': exag_weighted,
            f'{prefix}manipulative': urgency_weighted + pressure_weighted + exag_weighted,
            f'{prefix}combo_score': combo_score,
            f'{prefix}is_short': int(text_length < 100),
            f'{prefix}is_very_short': int(text_length < 50),
            f'{prefix}length_penalty': length_penalty,
            f'{prefix}short_risk': urgency_raw + money_raw + exag_raw if text_length < 100 else 0,
            f'{prefix}email': emails,
            f'{prefix}phone': phones,
            f'{prefix}url': urls,
            f'{prefix}contacts': emails + phones,
            f'{prefix}polarity': polarity,
            f'{prefix}subjectivity': subjectivity,
            f'{prefix}high_polarity': int(polarity > self.thresholds['polarity']),
            f'{prefix}high_subj': int(subjectivity > self.thresholds['subjectivity']),
        }

    def _empty_features(self, prefix):
        keys = ['length', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len',
                'caps_ratio', 'high_caps', 'exclaim', 'high_exclaim', 'question',
                'keyword', 'has_keyword',
                'urgency_raw', 'urgency', 'pressure_raw', 'pressure',
                'money_raw', 'money', 'exag_raw', 'exag',
                'manipulative', 'combo_score',
                'is_short', 'is_very_short', 'length_penalty', 'short_risk',
                'email', 'phone', 'url', 'contacts', 'polarity', 'subjectivity',
                'high_polarity', 'high_subj']
        return {f'{prefix}{k}': 0 for k in keys}

    def extract_company_features(self, company_profile):
        """회사 신뢰도"""
        if pd.isna(company_profile) or company_profile == '':
            return {'company_credibility': 0, 'has_awards': 0, 'has_partners': 0, 'has_year': 0}

        text = str(company_profile).lower()
        score = 0

        has_awards = int(any(w in text for w in ['award', 'certified', 'accredited']))
        score += has_awards * 0.3

        has_partners = int(any(w in text for w in ['partnership', 'partner with', 'collaboration']))
        score += has_partners * 0.25

        has_year = int(bool(re.search(r'\b(19|20)\d{2}\b', text)))
        score += has_year * 0.2

        score += min(len(company_profile) / 500, 1.0) * 0.25

        return {
            'company_credibility': score,
            'has_awards': has_awards,
            'has_partners': has_partners,
            'has_year': has_year
        }

    def extract_industry_risk(self, industry, function):
        """산업/직무 위험도"""
        ind_str = str(industry).lower().strip() if pd.notna(industry) else ''
        func_str = str(function).lower().strip() if pd.notna(function) else ''

        ind_risk = self.ind_risk.get(ind_str, self.overall_rate * 1.5 if ind_str == '' else self.overall_rate)
        func_risk = self.func_risk.get(func_str, self.overall_rate * 1.5 if func_str == '' else self.overall_rate)

        return {
            'ind_risk': ind_risk,
            'func_risk': func_risk,
            'combined_risk': (ind_risk + func_risk) / 2,
            'high_risk': int(ind_risk > self.overall_rate * 2 and func_risk > self.overall_rate * 2),
        }

    def extract_meta_features(self, row):
        """메타데이터"""
        weighted = [
            int(row.get('has_company_logo', 0)) * 3,
            int(pd.notna(row.get('salary_range'))) * 2,
            int(pd.notna(row.get('company_profile')) and row.get('company_profile') != '') * 2,
            int(pd.notna(row.get('requirements')) and row.get('requirements') != ''),
            int(pd.notna(row.get('benefits')) and row.get('benefits') != ''),
        ]
        completeness = sum(weighted) / 9.0

        return {
            'has_logo': int(row.get('has_company_logo', 0)),
            'has_salary': int(pd.notna(row.get('salary_range'))),
            'has_profile': int(pd.notna(row.get('company_profile')) and row.get('company_profile') != ''),
            'has_req': int(pd.notna(row.get('requirements')) and row.get('requirements') != ''),
            'has_benefits': int(pd.notna(row.get('benefits')) and row.get('benefits') != ''),
            'telecommute': int(row.get('telecommuting', 0)),
            'completeness': completeness,
            'low_info': int(completeness < 0.5),
        }

    def transform(self, df):
        """전체 변환"""
        features = []

        title_feat = df['title'].apply(lambda x: self.extract_text_features(x, 't_'))
        features.append(pd.DataFrame(list(title_feat)))

        desc_feat = df['description'].apply(lambda x: self.extract_text_features(x, 'd_'))
        features.append(pd.DataFrame(list(desc_feat)))

        req_feat = df['requirements'].apply(lambda x: self.extract_text_features(x, 'r_'))
        features.append(pd.DataFrame(list(req_feat)))

        comp_feat = df['company_profile'].apply(self.extract_company_features)
        features.append(pd.DataFrame(list(comp_feat)))

        ind_feat = df.apply(lambda row: self.extract_industry_risk(row.get('industry'), row.get('function')), axis=1)
        features.append(pd.DataFrame(list(ind_feat)))

        meta_feat = df.apply(self.extract_meta_features, axis=1)
        features.append(pd.DataFrame(list(meta_feat)))

        result = pd.concat(features, axis=1)

        # 상호작용 특성
        result['low_info_urgent'] = ((result['completeness'] < 0.3) & (result['d_urgency'] > 0)).astype(int)
        result['no_logo_money'] = ((result['has_logo'] == 0) & (result['d_money'] > 2)).astype(int)
        result['remote_high_subj'] = ((result['telecommute'] == 1) & (result['d_high_subj'] == 1)).astype(int)
        result['high_risk_low_info'] = (
                (result['ind_risk'] > result['ind_risk'].mean() * 2) & (result['completeness'] < 0.4)).astype(int)
        result['no_salary_exag'] = ((result['has_salary'] == 0) & (result['d_exag'] > 2)).astype(int)
        result['contact_urgent'] = ((result['d_contacts'] > 0) & (result['d_urgency'] > 0)).astype(int)
        result['short_urgent'] = ((result['d_is_short'] == 1) & (result['d_urgency_raw'] > 0)).astype(int)
        result['short_money'] = ((result['d_is_short'] == 1) & (result['d_money_raw'] > 1)).astype(int)
        result['short_exag'] = ((result['d_is_short'] == 1) & (result['d_exag_raw'] > 0)).astype(int)
        result['very_short_urgent'] = ((result['d_is_very_short'] == 1) & (result['d_urgency_raw'] > 0)).astype(int)
        result['title_urgent_money'] = ((result['t_urgency_raw'] > 0) & (result['t_money_raw'] > 0)).astype(int)
        result['title_short_urgent'] = ((result['t_is_short'] == 1) & (result['t_urgency_raw'] > 0)).astype(int)
        result['exclaim_low_info'] = ((result['d_exclaim'] > 3) & (result['completeness'] < 0.3)).astype(int)
        result['contacts_low_info'] = ((result['d_contacts'] > 0) & (result['completeness'] < 0.4)).astype(int)
        result['short_contacts'] = ((result['d_is_short'] == 1) & (result['d_contacts'] > 0)).astype(int)
        result['money_exag_combo'] = ((result['d_money_raw'] > 1) & (result['d_exag_raw'] > 1)).astype(int)
        result['triple_threat'] = (
                (result['d_urgency_raw'] > 0) & (result['d_money_raw'] > 0) & (result['d_exag_raw'] > 0)).astype(
            int)
        result['short_triple'] = ((result['d_is_short'] == 1) & (result['triple_threat'] == 1)).astype(int)

        return result


# ============================================================================
# BERTEmbedder
# ============================================================================

class BERTEmbedder:
    """BERT embedding 생성기"""

    def __init__(self, model_name='all-MiniLM-L6-v2', n_components=64):
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA

        self.model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_fitted = False
        self.n_components = n_components

    def transform(self, df, fit=False):
        """BERT embeddings 생성"""
        texts = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            desc = str(row.get('description', '')).strip()
            text = f"{title} [SEP] {desc}" if title and desc else (title or desc)
            texts.append(text if text else "empty")

        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

        if fit or not self.pca_fitted:
            embeddings_reduced = self.pca.fit_transform(embeddings)
            self.pca_fitted = True
        else:
            embeddings_reduced = self.pca.transform(embeddings)

        bert_df = pd.DataFrame(
            embeddings_reduced,
            columns=[f'bert_{i}' for i in range(self.n_components)],
            index=df.index
        )

        return bert_df


# ============================================================================
# FocalLossClassifier
# ============================================================================

class FocalLossClassifier:
    """Focal Loss Wrapper"""

    def __init__(self, base_model, alpha=0.25, gamma=2.0):
        self.base_model = base_model
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y):
        """학습"""
        return self.base_model.fit(X, y)

    def predict(self, X):
        """예측"""
        return self.base_model.predict(X)

    def predict_proba(self, X):
        """확률 예측"""
        return self.base_model.predict_proba(X)

    def __getattr__(self, name):
        """다른 속성은 base_model에 위임"""
        if name == 'base_model':
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'base_model'), name)

    def __getstate__(self):
        """pickle 직렬화"""
        return self.__dict__

    def __setstate__(self, state):
        """pickle 역직렬화"""
        self.__dict__.update(state)


# ============================================================================
# ProductionMonitor
# ============================================================================

class ProductionMonitor:
    """프로덕션 모니터링 (더미)"""

    def __init__(self):
        self.metrics = {}

    def update_performance_metrics(self, y_true, y_pred, y_proba):
        """성능 메트릭 업데이트"""
        pass

    def get_metrics(self):
        """메트릭 반환"""
        return self.metrics
