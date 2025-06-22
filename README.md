# LightRAG Terminal Q&A System

LightRAG를 활용한 터미널 기반 문서 Q&A 시스템입니다. 한국어와 영어 문서를 지원하며, OpenRouter, OpenAI, Ollama 등 다양한 LLM 서비스를 사용할 수 있습니다.

## 특징

- 🚀 **LightRAG 기반**: 빠르고 효율적인 검색 증강 생성
- 🌐 **다중 LLM 지원**: OpenRouter, OpenAI, Ollama
- 📄 **다양한 문서 형식**: TXT, PDF, DOCX, MD, XLSX
- 🇰🇷 **한국어/영어 지원**: 자동 언어 감지 및 처리
- 💻 **터미널 인터페이스**: 간편한 CLI 기반 상호작용

## 빠른 시작

### Windows

```bash
run.bat
```

### Linux/Mac

```bash
chmod +x run.sh
./run.sh
```

## 수동 설치

1. **Python 설치** (3.8 이상)

2. **가상환경 생성**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **패키지 설치**

```bash
pip install -r requirements.txt
```

4. **환경 설정**

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 설정
```

5. **실행**

```bash
python main.py chat
```

## 환경 설정

`.env` 파일에서 다음 설정을 구성하세요:

### Ollama 사용 (기본값, 무료)

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
```

### OpenAI 사용

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

### OpenRouter 사용

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

## 사용 방법

### 1. 문서 준비

`input/` 폴더에 분석할 문서를 넣으세요.

### 2. 대화형 모드 실행

```bash
python main.py chat
```

### 3. 명령어

- **질문하기**: 자연어로 질문 입력
- `/help`: 도움말 표시
- `/info`: 시스템 정보 표시
- `/reload`: 문서 다시 로드
- `exit` 또는 `q`: 종료

### CLI 명령어

```bash
# 대화형 모드
python main.py chat

# 문서만 로드
python main.py load

# 단일 질의
python main.py query "질문 내용"

# 시스템 정보
python main.py info
```

## 폴더 구조

```
anything-rag/
├── input/              # 문서 입력 폴더
├── rag_storage/        # LightRAG 인덱스 저장
├── logs/               # 로그 파일
├── src/
│   ├── Config/         # 설정 관리
│   ├── Service/        # 핵심 서비스
│   └── Utils/          # 유틸리티
├── .env               # 환경 설정
├── requirements.txt    # 의존성 패키지
├── main.py            # 메인 진입점
├── run.bat            # Windows 실행 스크립트
└── run.sh             # Linux/Mac 실행 스크립트
```

## 문제 해결

### Ollama 연결 실패

1. Ollama 설치: https://ollama.ai
2. 모델 다운로드: `ollama pull gemma3:1b`
3. Ollama 실행 확인: `ollama list`

### 패키지 설치 오류

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 인코딩 문제

- 한국어 문서는 UTF-8로 저장하세요
- 자동 인코딩 감지가 실패하면 UTF-8로 재저장

## 라이선스

MIT License

## 기반 프로젝트

- [LightRAG](https://github.com/HKUDS/LightRAG) - Simple and Fast Retrieval-Augmented Generation
