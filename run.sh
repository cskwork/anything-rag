#!/bin/bash

echo "========================================"
echo "LightRAG Terminal Q&A System"
echo "========================================"
echo ""

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3가 설치되어 있지 않습니다."
    echo "Python 3.8 이상을 설치해주세요."
    exit 1
fi

# 가상환경 확인 및 생성
if [ ! -d ".venv" ]; then
    echo "[1/4] 가상환경 생성 중..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] 가상환경 생성 실패"
        exit 1
    fi
else
    echo "[1/4] 기존 가상환경 사용"
fi

# 가상환경 활성화
echo "[2/4] 가상환경 활성화 중..."
source .venv/bin/activate

# pip 업그레이드
echo "[3/4] pip 업그레이드 중..."
pip install --upgrade pip > /dev/null 2>&1

# 패키지 설치
echo "[4/4] 필요한 패키지 설치 중..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] 패키지 설치 실패"
    echo "다음을 확인해주세요:"
    echo "- 인터넷 연결 상태"
    echo "- requirements.txt 파일 존재 여부"
    exit 1
fi

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo ""
    echo "[INFO] .env 파일이 없습니다. .env.example을 복사합니다."
    cp .env.example .env
    echo ""
    echo "========================================"
    echo "[중요] .env 파일을 편집해주세요:"
    echo "- OpenRouter, OpenAI, 또는 Ollama 설정"
    echo "- 기타 필요한 설정"
    echo "========================================"
    echo ""
    echo "계속하려면 Enter를 누르세요..."
    read
fi

# Ollama 확인 (선택사항)
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "[INFO] Ollama가 설치되어 있지 않습니다."
    echo "Ollama를 사용하려면 https://ollama.ai 에서 설치해주세요."
    echo "OpenRouter나 OpenAI를 사용한다면 무시하셔도 됩니다."
    echo ""
fi

# input 폴더 확인
if [ ! -d "input" ]; then
    mkdir -p input
    echo ""
    echo "[INFO] input 폴더가 생성되었습니다."
    echo "분석할 문서를 input 폴더에 넣어주세요."
    echo "지원 형식: .txt, .pdf, .docx, .md, .xlsx"
    echo ""
fi

echo ""
echo "========================================"
echo "설치 완료! LightRAG 시스템을 시작합니다."
echo "========================================"
echo ""

# 메인 프로그램 실행
python main.py chat

# 가상환경 비활성화
deactivate 