"""LightRAG 기반 RAG 서비스"""
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
from lightrag import LightRAG
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.llm import openai_complete_if_cache, openai_embedding
from src.Config.config import settings
from src.Service.llm_service import get_llm_service
from src.Service.document_loader import DocumentLoader


class RAGService:
    """LightRAG를 활용한 RAG 서비스"""
    
    def __init__(self):
        self.working_dir = str(settings.lightrag_working_dir)
        self.chunk_size = settings.lightrag_chunk_size
        self.chunk_overlap = settings.lightrag_chunk_overlap
        self.llm_service_type = settings.get_llm_service()
        self.rag = None
        self.document_loader = DocumentLoader()
        self._initialize_rag()
    
    def _initialize_rag(self):
        """LightRAG 초기화"""
        try:
            # 디렉토리 생성
            settings.create_directories()
            
            # LLM 서비스에 따른 설정
            if self.llm_service_type == "ollama":
                self.rag = LightRAG(
                    working_dir=self.working_dir,
                    llm_model_func=self._ollama_complete,
                    embedding_func=self._ollama_embedding,
                    chunk_token_size=self.chunk_size,
                    chunk_overlap_token_size=self.chunk_overlap,
                )
                logger.info(f"LightRAG 초기화 완료 (Ollama: {settings.ollama_model})")
            
            elif self.llm_service_type == "openai":
                self.rag = LightRAG(
                    working_dir=self.working_dir,
                    llm_model_func=openai_complete_if_cache,
                    llm_model_name=settings.openai_model,
                    embedding_func=openai_embedding,
                    embedding_model=settings.lightrag_embedding_model,
                    chunk_token_size=self.chunk_size,
                    chunk_overlap_token_size=self.chunk_overlap,
                    api_key=settings.openai_api_key,
                )
                logger.info(f"LightRAG 초기화 완료 (OpenAI: {settings.openai_model})")
            
            elif self.llm_service_type == "openrouter":
                # OpenRouter는 커스텀 함수 사용
                self.rag = LightRAG(
                    working_dir=self.working_dir,
                    llm_model_func=self._openrouter_complete,
                    embedding_func=self._openrouter_embedding,
                    chunk_token_size=self.chunk_size,
                    chunk_overlap_token_size=self.chunk_overlap,
                )
                logger.info(f"LightRAG 초기화 완료 (OpenRouter: {settings.openrouter_model})")
            
        except Exception as e:
            logger.error(f"LightRAG 초기화 실패: {e}")
            raise
    
    async def _ollama_complete(self, prompt: str, **kwargs) -> str:
        """Ollama 완성 함수"""
        return await ollama_model_complete(
            prompt,
            model_name=settings.ollama_model,
            host=settings.ollama_host,
            **kwargs
        )
    
    async def _ollama_embedding(self, text: str, **kwargs) -> List[float]:
        """Ollama 임베딩 함수"""
        return await ollama_embedding(
            text,
            model_name=settings.ollama_model,
            host=settings.ollama_host,
            **kwargs
        )
    
    async def _openrouter_complete(self, prompt: str, **kwargs) -> str:
        """OpenRouter 완성 함수"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', settings.temperature),
            max_tokens=kwargs.get('max_tokens', settings.max_tokens),
        )
        return response.choices[0].message.content
    
    async def _openrouter_embedding(self, text: str, **kwargs) -> List[float]:
        """OpenRouter 임베딩 함수 (OpenAI 사용)"""
        if settings.openai_api_key:
            return await openai_embedding(
                text,
                model=settings.lightrag_embedding_model,
                api_key=settings.openai_api_key,
                **kwargs
            )
        else:
            # 간단한 더미 임베딩
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            return [float(b) / 255.0 for b in hash_bytes[:384]]
    
    async def insert_documents(self, documents: List[Dict[str, str]] = None):
        """문서를 RAG에 삽입"""
        try:
            if documents is None:
                # input 폴더에서 문서 로드
                documents = self.document_loader.load_documents()
            
            if not documents:
                logger.warning("삽입할 문서가 없습니다.")
                return
            
            # 문서 내용만 추출
            contents = [doc['content'] for doc in documents]
            
            # 배치로 삽입
            logger.info(f"{len(documents)}개 문서 삽입 시작...")
            await self.rag.ainsert(contents)
            logger.info("문서 삽입 완료")
            
        except Exception as e:
            logger.error(f"문서 삽입 실패: {e}")
            raise
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """RAG에 질의"""
        try:
            # mode: naive, local, global, hybrid
            logger.debug(f"질의: {question} (모드: {mode})")
            
            # 한국어 질의인 경우 프롬프트 개선
            if self._is_korean(question):
                question = f"다음 질문에 대해 한국어로 답변해주세요: {question}"
            
            response = await self.rag.aquery(question, param={"mode": mode})
            return response
            
        except Exception as e:
            logger.error(f"질의 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def _is_korean(self, text: str) -> bool:
        """텍스트가 한국어인지 확인"""
        korean_chars = sum(1 for char in text if '가' <= char <= '힣')
        return korean_chars / len(text) > 0.3 if text else False
    
    async def get_indexed_info(self) -> Dict[str, any]:
        """인덱싱된 정보 반환"""
        try:
            # 저장된 파일 확인
            storage_path = Path(self.working_dir)
            info = {
                'working_dir': str(storage_path),
                'exists': storage_path.exists(),
                'files': []
            }
            
            if storage_path.exists():
                for file in storage_path.iterdir():
                    if file.is_file():
                        info['files'].append({
                            'name': file.name,
                            'size': file.stat().st_size
                        })
            
            return info
        except Exception as e:
            logger.error(f"인덱스 정보 조회 실패: {e}")
            return {'error': str(e)} 