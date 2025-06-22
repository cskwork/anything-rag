"""설정 관리 모듈"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # LLM 서비스 설정
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    openrouter_model: str = Field(default="anthropic/claude-3-haiku", env="OPENROUTER_MODEL")
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="gemma3:1b", env="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(default="bge-m3:latest", env="OLLAMA_EMBEDDING_MODEL")
    
    # LightRAG 설정
    lightrag_working_dir: Path = Field(default=Path("./rag_storage"), env="LIGHTRAG_WORKING_DIR")
    lightrag_chunk_size: int = Field(default=1200, env="LIGHTRAG_CHUNK_SIZE")
    lightrag_chunk_overlap: int = Field(default=100, env="LIGHTRAG_CHUNK_OVERLAP")
    lightrag_embedding_model: str = Field(default="text-embedding-3-small", env="LIGHTRAG_EMBEDDING_MODEL")
    
    # 시스템 설정
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    language: str = Field(default="ko", env="LANGUAGE")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    embedding_dim: Optional[int] = Field(default=None, env="EMBEDDING_DIM")
    
    # 문서 처리 설정
    supported_extensions_str: str = Field(
        alias="SUPPORTED_EXTENSIONS",
        default=".txt,.pdf,.docx,.md,.xlsx"
    )

    @computed_field
    @property
    def supported_extensions(self) -> List[str]:
        """
        SUPPORTED_EXTENSIONS 환경 변수를 파싱하여 리스트로 변환합니다.
        콤마로 구분된 문자열 (e.g., ".txt,.pdf") 또는 JSON 배열 (e.g., '[\".txt\", \".pdf\"]') 형식을 지원합니다.
        """
        value = self.supported_extensions_str.strip()
        
        if not value:
            return [".txt", ".pdf", ".docx", ".md", ".xlsx"]
        
        if value.startswith('[') and value.endswith(']'):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return [ext.strip() for ext in value.split(',') if ext.strip()]

    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    encoding: str = Field(default="utf-8", env="ENCODING")
    
    # 디렉토리 설정
    input_dir: Path = Field(default=Path("./input"), env="INPUT_DIR")
    backup_dir: Path = Field(default=Path("./backup"), env="BACKUP_DIR")
    logs_dir: Path = Field(default=Path("./logs"), env="LOGS_DIR")
    
    # LLM 제공자 선택 (auto, ollama, openai, openrouter)
    llm_provider: str = Field(default="auto", env="LLM_PROVIDER")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    def get_llm_service(self) -> str:
        """사용 가능한 LLM 서비스 확인"""
        if self.openrouter_api_key:
            return "openrouter"
        elif self.openai_api_key:
            return "openai"
        else:
            return "ollama"
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        self.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

# 전역 설정 인스턴스
settings = Settings()