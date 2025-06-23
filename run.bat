@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo LightRAG Terminal Q^&A System
echo ========================================
echo.

:: Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되어 있지 않습니다.
    echo Python 3.8 이상을 설치해주세요: https://www.python.org
    pause
    exit /b 1
)

:: 가상환경 확인 및 생성
if not exist ".venv" (
    echo [1/4] 가상환경 생성 중...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] 가상환경 생성 실패
        pause
        exit /b 1
    )
) else (
    echo [1/4] 기존 가상환경 사용
)

:: 가상환경 활성화
echo [2/4] 가상환경 활성화 중...
call .venv\Scripts\activate.bat

:: pip 업그레이드
echo [3/4] pip 업그레이드 중...
python -m pip install --upgrade pip >nul 2>&1

:: 패키지 설치
echo [4/4] 필요한 패키지 설치 중...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] 패키지 설치 실패
    echo 다음을 확인해주세요:
    echo - 인터넷 연결 상태
    echo - requirements.txt 파일 존재 여부
    pause
    exit /b 1
)

:: .env 파일 확인
if not exist ".env" (
    echo.
    echo [INFO] .env 파일이 없습니다. .env.example을 복사합니다.
    copy .env.example .env >nul 2>&1
    echo.
    echo ========================================
    echo [중요] .env 파일을 편집해주세요:
    echo - OpenRouter, OpenAI, 또는 Ollama 설정
    echo - 기타 필요한 설정
    echo ========================================
    echo.
    pause
)

:: Ollama 확인 (선택사항)
ollama list >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Ollama가 설치되어 있지 않습니다.
    echo Ollama를 사용하려면 https://ollama.ai 에서 설치해주세요.
    echo OpenRouter나 OpenAI를 사용한다면 무시하셔도 됩니다.
    echo.
)

:: input 폴더 확인
if not exist "input" (
    mkdir input
    echo.
    echo [INFO] input 폴더가 생성되었습니다.
    echo 분석할 문서를 input 폴더에 넣어주세요.
    echo 지원 형식: .txt, .pdf, .docx, .md, .xlsx
    echo.
)

echo.
echo ========================================
echo 설치 완료! LightRAG 시스템을 시작합니다.
echo ========================================
echo.

REM Q&A 시스템 실행
REM 스크립트에 전달된 인자를 python main.py로 전달
IF "%1"=="" (
    REM 인자가 없으면 기본 'chat' 명령어 실행
    python main.py chat
) ELSE (
    REM 인자가 있으면 그대로 전달
    python main.py %*
)

:: 가상환경 비활성화
deactivate

pause 