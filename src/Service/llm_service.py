"""LLM 서비스 관리 모듈"""
import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import ollama
from openai import OpenAI
from loguru import logger
from src.Config.config import settings


class LLMService(ABC):
    """LLM 서비스 추상 클래스"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> list:
        """텍스트 임베딩"""
        pass


class OllamaService(LLMService):
    """Ollama 서비스"""
    
    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model
        self._check_and_suggest_model()
    
    def _check_and_suggest_model(self):
        """설치된 모델 확인 및 제안"""
        try:
            models = self.client.list()
            installed_models = [model['name'] for model in models.get('models', [])]
            
            if installed_models:
                logger.info(f"설치된 Ollama 모델: {', '.join(installed_models)}")
                
                # 설정된 모델이 없으면 제안
                if self.model not in installed_models:
                    if 'gemma3:1b' in installed_models:
                        self.model = 'gemma3:1b'
                    else:
                        self.model = installed_models[0]
                    logger.info(f"모델을 {self.model}로 설정합니다.")
            else:
                logger.warning("설치된 Ollama 모델이 없습니다. 'ollama pull gemma3:1b'를 실행해주세요.")
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': kwargs.get('temperature', settings.temperature),
                    'max_tokens': kwargs.get('max_tokens', settings.max_tokens),
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama 생성 오류: {e}")
            raise
    
    async def embed(self, text: str) -> list:
        """텍스트 임베딩"""
        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Ollama 임베딩 오류: {e}")
            raise


class OpenAIService(LLMService):
    """OpenAI 서비스"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.lightrag_embedding_model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', settings.temperature),
                max_tokens=kwargs.get('max_tokens', settings.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 생성 오류: {e}")
            raise
    
    async def embed(self, text: str) -> list:
        """텍스트 임베딩"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI 임베딩 오류: {e}")
            raise


class OpenRouterService(LLMService):
    """OpenRouter 서비스"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        self.model = settings.openrouter_model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', settings.temperature),
                max_tokens=kwargs.get('max_tokens', settings.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter 생성 오류: {e}")
            raise
    
    async def embed(self, text: str) -> list:
        """텍스트 임베딩 - OpenRouter는 임베딩을 지원하지 않으므로 OpenAI 사용"""
        if settings.openai_api_key:
            openai_service = OpenAIService()
            return await openai_service.embed(text)
        else:
            logger.warning("임베딩을 위해 OpenAI API 키가 필요합니다.")
            # 간단한 해시 기반 임베딩 (실제로는 사용하지 않는 것이 좋음)
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            return [float(b) / 255.0 for b in hash_bytes[:384]]  # 384차원


def get_llm_service() -> LLMService:
    """설정에 따라 적절한 LLM 서비스 반환"""
    service_type = settings.get_llm_service()
    
    if service_type == "openrouter":
        logger.info("OpenRouter 서비스 사용")
        return OpenRouterService()
    elif service_type == "openai":
        logger.info("OpenAI 서비스 사용")
        return OpenAIService()
    else:
        logger.info("Ollama 서비스 사용 (기본값)")
        return OllamaService() 