# LightRAG Terminal Q&A System

LightRAG를 활용한 터미널 기반 문서 Q&A 시스템입니다. 한국어와 영어 문서를 지원하며, 로컬 LLM API, OpenRouter, OpenAI, Ollama 등 다양한 LLM 서비스를 사용할 수 있습니다.

## ✨ 특징

- 🚀 **LightRAG 기반**: 빠르고 효율적인 검색 증강 생성 (Retrieval-Augmented Generation)
- 🌐 **다중 LLM 지원**: 로컬 LLM API, OpenRouter, OpenAI, Ollama 자동 감지 및 선택
- 📄 **다양한 문서 형식**: TXT, PDF, DOCX, MD, XLSX 지원
- 🇰🇷 **한국어/영어 지원**: 자동 언어 감지 및 처리
- 💻 **풍부한 터미널 UI**: Rich 라이브러리 기반 아름다운 인터페이스
- 🔄 **스마트 문서 관리**: 파일 변경 감지 (MD5 해시) 및 증분 업데이트
- 📝 **대화 히스토리**: 컨텍스트 유지 대화
- 🛠️ **유연한 설정**: 환경 변수 기반 세부 설정

## 🚀 빠른 시작

### Windows

```bash
run.bat
```

### Linux/Mac

```bash
chmod +x run.sh
./run.sh
```

## 📦 수동 설치

1. **Python 설치** (3.8 이상 권장)

2. **가상환경 생성 및 활성화**

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
# .env 파일 생성 및 편집
cp .env.example .env  # 또는 수동으로 .env 파일 생성
```

5. **문서 준비**

```bash
# input/ 폴더에 분석할 문서 넣기
mkdir -p input
# 여기에 .txt, .pdf, .docx, .md, .xlsx 파일들을 넣으세요
```

6. **실행**

```bash
python main.py chat
```

## ⚙️ 환경 설정

`.env` 파일에서 다음 설정을 구성하세요:

### LLM 제공자 선택

시스템은 다음 우선순위로 LLM 서비스를 자동 선택합니다:
**로컬 LLM API → Ollama → OpenRouter**

```env
# LLM 제공자 선택 - 우선순위: local → ollama → openrouter  
# Options: auto, local, ollama, openai, openrouter
LLM_PROVIDER=auto
```

### 로컬 LLM API 사용 (최우선, 무료)

```env
# 로컬 LLM API 설정 (OpenAI 호환)
LOCAL_API_HOST=http://localhost:3284
LLM_PROVIDER=local  # 또는 auto (자동 감지)

# 임베딩은 로컬 API → Ollama 폴백
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
```

### Ollama 사용 (기본 폴백, 로컬 무료)

```env
# Ollama 설정
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
LLM_PROVIDER=ollama  # 또는 auto (자동 감지)

# 시스템 설정
LOG_LEVEL=INFO
LANGUAGE=ko
```

### OpenAI 사용

```env
# OpenAI 설정
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
LLM_PROVIDER=openai  # 또는 auto (자동 감지)

# LightRAG 임베딩 모델
LIGHTRAG_EMBEDDING_MODEL=text-embedding-3-small
```

### OpenRouter 사용

```env
# OpenRouter 설정
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=anthropic/claude-3-haiku
LLM_PROVIDER=openrouter  # 또는 auto (자동 감지)

# 또는 다른 모델들
# OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
# OPENROUTER_MODEL=openai/gpt-3.5-turbo
```

### 고급 설정

```env
# LightRAG 설정
LIGHTRAG_WORKING_DIR=./rag_storage
LIGHTRAG_CHUNK_SIZE=1200
LIGHTRAG_CHUNK_OVERLAP=100
LIGHTRAG_HISTORY_TURNS=3

# 문서 처리 설정
SUPPORTED_EXTENSIONS=.txt,.pdf,.docx,.md,.xlsx
MAX_FILE_SIZE_MB=50
ENCODING=utf-8

# LLM 매개변수
MAX_TOKENS=2000
TEMPERATURE=0.7

# 디렉토리 설정
INPUT_DIR=./input
BACKUP_DIR=./backup
LOGS_DIR=./logs
```

## 🎯 사용 방법

### 기본 사용 흐름

1. **문서 준비**: `input/` 폴더에 분석할 문서들을 넣기
2. **대화형 모드 실행**: `python main.py chat`
3. **질문하기**: 자연어로 질문 입력
4. **특수 명령어**: 시스템 관리 명령어 활용

### CLI 명령어

```bash
# 대화형 모드 (권장)
python main.py chat

# 문서 로드만 수행
python main.py load

# 모든 문서 강제 재로드
python main.py load --force-reload

# 단일 질의
python main.py query "LightRAG의 주요 특징은 무엇인가요?"

# 질의 모드 지정
python main.py query "질문" --mode hybrid

# 시스템 정보 확인
python main.py info

# 시스템 구성 요소 테스트
python main.py test
```

### 대화형 모드 특수 명령어

대화형 모드에서 다음 명령어들을 사용할 수 있습니다:

- `/help` - 사용 가능한 명령어 목록 표시
- `/info` - 현재 시스템 상태 및 인덱스 정보
- `/reload` - 신규/변경된 문서만 다시 로드
- `/reload-all` - 모든 문서 강제 재로드
- `/reset` - 임베딩 상태 초기화 (다음 로드시 모든 파일 재처리)
- `exit`, `quit`, `q` - 프로그램 종료

### 질의 모드

LightRAG는 4가지 질의 모드를 지원합니다:

- **hybrid** (기본값): 로컬과 글로벌 검색을 결합한 최적의 방식
- **naive**: 단순 벡터 유사도 검색
- **local**: 로컬 지식 그래프 기반 검색
- **global**: 글로벌 지식 그래프 기반 검색

## 💡 실제 사용 예시

### 첫 실행 과정

```bash
# 1. 프로젝트 클론 또는 다운로드
git clone <repository-url>
cd anything-rag

# 2. 환경 설정
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경 변수 설정 (.env 파일 생성)
echo "LLM_PROVIDER=auto" > .env
echo "OLLAMA_HOST=http://localhost:11434" >> .env
echo "OLLAMA_MODEL=gemma3:1b" >> .env
echo "LOG_LEVEL=INFO" >> .env

# 5. 문서 준비
mkdir -p input
cp your_documents.pdf input/
cp your_text_files.txt input/

# 6. 시스템 테스트
python main.py test

# 7. 대화형 모드 실행
python main.py chat
```

### 대화형 모드 사용 예시

```
# 시스템 시작
python main.py chat

# 문서 자동 로드 후 질문하기
질문: LightRAG의 주요 특징은 무엇인가요?
답변: [문서 기반 상세 답변]

# 특수 명령어 사용
질문: /info
답변: [시스템 정보 표시]

질문: /help
답변: [도움말 표시]

# 새 문서 추가 후 리로드
질문: /reload
답변: [신규 문서만 로드]

# 대화 맥락 활용
질문: 앞서 말한 특징 중 가장 중요한 것은?
답변: [이전 답변을 참고한 답변]

# 종료
질문: exit
```

### CLI 단일 명령어 사용 예시

```bash
# 문서 로드만 수행
python main.py load
# 출력: "문서 로드 완료! 5개 파일 처리됨"

# 강제 전체 재로드
python main.py load --force-reload
# 출력: "모든 문서 재처리 완료!"

# 단일 질의 (대화형 모드 없이)
python main.py query "프로젝트의 주요 기능은?"
# 출력: [즉시 답변 반환]

# 특정 모드로 질의
python main.py query "성능 최적화 방법은?" --mode local
# 출력: [로컬 지식 그래프 기반 답변]

# 시스템 정보 확인
python main.py info
# 출력: [설정 정보 테이블]
```

### 다양한 질문 유형 예시

**기본 정보 질의:**
```
질문: "이 문서에서 다루는 주요 주제는 무엇인가요?"
질문: "프로젝트의 목적과 목표를 설명해주세요"
질문: "주요 기능들을 나열해주세요"
```

**구체적 정보 검색:**
```
질문: "설치 방법을 단계별로 알려주세요"
질문: "오류 해결 방법을 찾아주세요"
질문: "성능 최적화 팁이 있나요?"
```

**비교 및 분석:**
```
질문: "로컬 LLM API와 Ollama의 장단점을 비교해주세요"
질문: "어떤 LLM 서비스를 선택해야 할까요?"
질문: "각 질의 모드의 차이점은 무엇인가요?"
```

**맥락 기반 대화:**
```
질문: "LightRAG의 특징을 설명해주세요"
답변: [LightRAG 특징 설명]
질문: "그 중에서 가장 중요한 특징은 무엇인가요?"
답변: [이전 답변을 참고한 구체적 설명]
```

### 문제 상황별 대처법

**새 문서 추가했을 때:**
```bash
# 방법 1: 대화형 모드에서
질문: /reload

# 방법 2: CLI에서
python main.py load
```

**전체 재처리가 필요할 때:**
```bash
# 방법 1: 대화형 모드에서
질문: /reload-all

# 방법 2: CLI에서
python main.py load --force-reload
```

**임베딩 상태 초기화:**
```bash
# 대화형 모드에서
질문: /reset
질문: y  # 확인

# 그 다음 재로드
질문: /reload-all
```

**시스템 문제 진단:**
```bash
# 1. 시스템 테스트
python main.py test

# 2. 설정 확인
python main.py info

# 3. 로그 확인
tail -f logs/app.log
```

## 📁 프로젝트 구조

```
anything-rag/
├── 📂 input/              # 📄 문서 입력 폴더 (여기에 파일을 넣으세요)
├── 📂 rag_storage/        # 🗄️ LightRAG 인덱스 및 지식 그래프 저장소
├── 📂 logs/               # 📋 로그 파일들
├── 📂 backup/             # 💾 백업 파일들
├── 📂 src/
│   ├── 📂 Config/         # ⚙️ 설정 관리
│   │   ├── __init__.py
│   │   └── config.py      # 환경 변수 및 설정 클래스
│   ├── 📂 Service/        # 🔧 핵심 서비스 로직
│   │   ├── __init__.py
│   │   ├── document_loader.py    # 문서 로딩 및 변경 감지
│   │   ├── llm_service.py        # LLM 서비스 추상화 (로컬/Ollama/OpenAI/OpenRouter)
│   │   ├── local_api_service.py  # 로컬 메시지 API 서비스
│   │   └── rag_service.py        # RAG 메인 서비스
│   ├── 📂 Utils/          # 🛠️ 유틸리티 함수들
│   │   ├── __init__.py
│   │   └── file_tracker.py       # 파일 변경 추적
│   └── 📂 test/           # 🧪 테스트 코드
│       ├── __init__.py
│       ├── test_document_loader.py
│       ├── test_llm_service.py
│       └── test_rag_service.py
├── 📄 .env                # 🔐 환경 설정 (사용자가 생성)
├── 📄 .env.example        # 📋 환경 설정 예시
├── 📄 requirements.txt    # 📦 Python 패키지 의존성
├── 📄 main.py            # 🚪 메인 진입점 (CLI)
├── 📄 run.bat            # 🪟 Windows 실행 스크립트
├── 📄 run.sh             # 🐧 Linux/Mac 실행 스크립트
└── 📄 README.md          # 📚 이 파일
```

## 🔧 문제 해결

### LLM 서비스 관련

**로컬 LLM API 연결 실패 시:**
1. 로컬 LLM 서버 상태 확인: `curl http://localhost:3284/health`
2. OpenAI 호환 API 엔드포인트 확인: `/v1/chat/completions`, `/v1/embeddings`
3. 서버 설정에서 포트 3284 허용 여부 확인
4. 폴백으로 Ollama 사용됨

**Ollama 연결 실패 시:**
1. Ollama 설치: https://ollama.ai
2. 모델 다운로드: `ollama pull gemma3:1b`
3. 임베딩 모델: `ollama pull bge-m3:latest`
4. 서비스 실행 확인: `ollama list`

**모델 추천:**
- 경량: `gemma3:1b` (1B 파라미터)
- 균형: `llama3.2:3b` (3B 파라미터) 
- 고성능: `llama3.1:8b` (8B 파라미터)

### 패키지 설치 문제

```bash
# 패키지 캐시 클리어 후 재설치
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Windows에서 특정 패키지 실패 시
pip install --upgrade setuptools wheel
```

### 문서 처리 문제

**인코딩 오류:**
- 한국어 문서는 UTF-8로 저장
- 자동 인코딩 감지 실패 시 `ENCODING=utf-8` 설정

**지원되지 않는 파일:**
- `SUPPORTED_EXTENSIONS` 환경 변수로 확장자 추가 가능
- 예: `SUPPORTED_EXTENSIONS=.txt,.pdf,.docx,.md,.xlsx,.html`

**메모리 부족:**
- `LIGHTRAG_CHUNK_SIZE` 값을 줄이기 (기본값: 1200)
- `MAX_FILE_SIZE_MB` 값으로 파일 크기 제한 (기본값: 50MB)

### 성능 최적화

**임베딩 속도 향상:**
- Ollama 사용 시 GPU 가속 활성화
- `LIGHTRAG_CHUNK_SIZE` 조정으로 처리량 최적화

**메모리 사용량 감소:**
- `LIGHTRAG_CHUNK_OVERLAP` 값 감소
- 불필요한 파일 `backup/` 폴더로 이동

## 📊 주요 의존성

- **lightrag-hku**: 핵심 RAG 엔진
- **typer**: CLI 인터페이스
- **rich**: 터미널 UI
- **loguru**: 로깅 시스템
- **pydantic**: 설정 관리
- **ollama/openai**: LLM 서비스 클라이언트

## 🔍 시스템 테스트

시스템이 올바르게 설정되었는지 확인:

```bash
python main.py test
```

이 명령어는 다음을 검증합니다:
- LLM 서비스 연결
- 임베딩 서비스 동작
- RAG 서비스 초기화
- 문서 로드 기능

## 🤝 기여하기

1. 이슈 리포트: 버그나 개선사항을 GitHub Issues에 등록
2. 코드 기여: Pull Request를 통한 기능 추가나 버그 수정
3. 문서 개선: README나 코드 주석 개선

## 📄 라이선스

MIT License

## 🙏 기반 프로젝트

- [LightRAG](https://github.com/HKUDS/LightRAG) - Simple and Fast Retrieval-Augmented Generation
- [Typer](https://typer.tiangolo.com/) - Modern CLI framework
- [Rich](https://rich.readthedocs.io/) - Rich text and beautiful formatting

---

**Glossary (용어 설명)**
- **RAG**: 검색 증강 생성. 문서에서 관련 정보를 찾아 더 정확한 답변을 생성하는 AI 기술
- **임베딩**: 텍스트를 숫자 벡터로 변환하여 컴퓨터가 의미를 이해할 수 있게 하는 과정
- **청크**: 긴 문서를 작은 단위로 나눈 조각. 검색과 처리를 효율적으로 하기 위함
- **LLM**: Large Language Model. GPT, Claude 같은 대규모 언어 모델
- **지식 그래프**: 정보들 간의 관계를 네트워크 형태로 표현한 데이터 구조
