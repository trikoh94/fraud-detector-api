FROM python:3.11-slim

# 필수 시스템 라이브러리
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . .

# 환경변수
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 서버 시작
CMD ["python", "main.py"]