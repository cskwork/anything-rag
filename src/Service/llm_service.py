"""LLM 서비스 관리 모듈"""
import os
import aiohttp
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import ollama
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from src.Config.config import settings

# 모델별 임베딩 차원 정보
OLLAMA_EMBEDDING_DIMS = {
    "bge-m3": 1024,
    "bge-m3:latest": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "gemma:2b": 2048, # 예시, 실제 gemma 모델은 임베딩 전용이 아닐 수 있음
}

OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
}


class LLMService(ABC):
    """LLM 서비스 추상 클래스"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    async def embed(self, texts) -> list:
        """텍스트 임베딩"""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """임베딩 벡터의 차원"""
        pass


class OllamaService(LLMService):
    """Ollama 서비스"""
    
    def __init__(self, client: ollama.AsyncClient, chat_model: str, embedding_model: str):
        self.client = client
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    @classmethod
    async def create(cls) -> "OllamaService":
        """비동기 Ollama 서비스 생성자"""
        client = ollama.AsyncClient(host=settings.ollama_host)
        chat_model = settings.ollama_model
        embedding_model = settings.ollama_embedding_model
        await cls._check_and_suggest_models(client, chat_model, embedding_model)
        return cls(client, chat_model, embedding_model)

    @staticmethod
    async def _check_and_suggest_models(client: ollama.AsyncClient, chat_model: str, embedding_model: str):
        """설치된 모델 확인 및 제안"""
        try:
            response = await client.list()
            logger.debug(f"Ollama raw list response: {response}")

            model_list = response.get('models', [])
            installed_models = [model_info.get('name') for model_info in model_list if isinstance(model_info, dict)]

            if installed_models:
                logger.info(f"Ollama에서 다음 모델을 성공적으로 인식했습니다: {', '.join(installed_models)}")
                
                # 채팅 모델 선택 (안정성 우선순위)
                if chat_model not in installed_models:
                    # 안정적인 모델 우선순위: gemma3:1b > deepseek-r1:1.5b > gemma3:latest (리소스 고려)
                    preferred_models = ['gemma3:1b', 'deepseek-r1:1.5b', 'gemma3:latest']
                    new_model = None
                    for preferred in preferred_models:
                        if preferred in installed_models:
                            new_model = preferred
                            break
                    
                    if not new_model:
                        new_model = installed_models[0]
                    
                    logger.info(f"설정된 채팅 모델({chat_model})을 찾을 수 없어 {new_model}(으)로 자동 설정합니다.")
                    settings.ollama_model = new_model
                
                # 임베딩 모델 선택
                if embedding_model not in installed_models:
                    # 임베딩 전용 모델 우선, 없으면 채팅 모델 사용
                    embedding_models = ['bge-m3:latest', 'nomic-embed-text', 'all-minilm']
                    new_embedding_model = None
                    for emb_model in embedding_models:
                        if emb_model in installed_models:
                            new_embedding_model = emb_model
                            break
                    
                    if not new_embedding_model:
                        new_embedding_model = settings.ollama_model  # 채팅 모델 사용
                    
                    logger.info(f"설정된 임베딩 모델({embedding_model})을 찾을 수 없어 {new_embedding_model}(으)로 자동 설정합니다.")
                    settings.ollama_embedding_model = new_embedding_model
            else:
                logger.warning("Ollama 서버 응답에서 설치된 모델 정보를 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"Ollama와 통신 중 오류 발생: {e}", exc_info=True)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            # 메모리 부족 방지를 위한 보수적인 설정
            options = {
                'temperature': kwargs.get('temperature', settings.temperature),
                'max_tokens': min(kwargs.get('max_tokens', settings.max_tokens), 1000),  # 최대 토큰 제한
                'num_ctx': 8192,  # 컨텍스트 크기 제한
                'num_predict': 512,  # 예측 토큰 제한
                'top_k': 40,
                'top_p': 0.9,
            }
            
            response = await self.client.generate(
                model=self.chat_model,
                prompt=prompt,
                options=options
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama 생성 오류: {e}")
            raise
    
    async def embed(self, texts) -> list:
        """텍스트 임베딩 - LightRAG 호환성"""
        try:
            import numpy as np
            
            # LightRAG는 단일 텍스트나 텍스트 리스트를 전달할 수 있음
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                texts = [str(texts)]
            
            # 빈 리스트 처리
            if not texts:
                logger.warning("빈 텍스트 리스트에 대한 임베딩 요청")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            results = []
            for text in texts:
                # 각 텍스트 처리
                if not text or not str(text).strip():
                    logger.warning("빈 텍스트에 대한 임베딩, 더미 벡터 사용")
                    embedding = [0.0] * self.embedding_dim
                else:
                    try:
                        response = await self.client.embeddings(
                            model=self.embedding_model,
                            prompt=str(text).strip()
                        )
                        
                        embedding = response.get('embedding', [])
                        if not embedding or len(embedding) == 0:
                            logger.warning("Ollama로부터 빈 임베딩 응답, 더미 벡터 사용")
                            embedding = [0.0] * self.embedding_dim
                        elif len(embedding) != self.embedding_dim:
                            logger.warning(f"임베딩 차원 불일치: 반환={len(embedding)}, 예상={self.embedding_dim}")
                            # 차원 조정
                            if len(embedding) > self.embedding_dim:
                                embedding = embedding[:self.embedding_dim]
                            else:
                                embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                    except Exception as embed_error:
                        logger.error(f"개별 텍스트 임베딩 실패: {embed_error}")
                        embedding = [0.0] * self.embedding_dim
                
                results.append(embedding)
            
            # 단일 텍스트인 경우 첫 번째 결과만 반환
            if len(texts) == 1:
                return results[0]
            
            # numpy 배열로 변환하여 차원 확인
            result_array = np.array(results)
            if result_array.shape[0] == 0:
                logger.warning("결과 배열이 비어있음, 더미 벡터 반환")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"Ollama 임베딩 처리 중 오류: {e}")
            # 안전한 더미 벡터 반환
            if isinstance(texts, list) and len(texts) > 1:
                return [[0.0] * self.embedding_dim for _ in texts]
            else:
                return [0.0] * self.embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Ollama 모델의 임베딩 차원을 반환합니다."""
        if settings.embedding_dim:
            return settings.embedding_dim
        for model_key, dim in OLLAMA_EMBEDDING_DIMS.items():
            if model_key in self.embedding_model:
                logger.debug(f"{self.embedding_model} 모델의 임베딩 차원을 {dim}(으)로 감지했습니다.")
                return dim
        logger.warning(f"Ollama 임베딩 모델({self.embedding_model})의 차원을 알 수 없습니다. .env에 EMBEDDING_DIM=값 을 설정해주세요. 기본값 768을 사용합니다.")
        return 768


class OpenAIService(LLMService):
    """OpenAI 서비스"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.lightrag_embedding_model
    
    @classmethod
    async def create(cls) -> "OpenAIService":
        # 현재 생성자는 I/O가 없으므로 간단히 인스턴스화
        return cls()

    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', settings.temperature),
                max_tokens=kwargs.get('max_tokens', settings.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 생성 오류: {e}")
            raise
    
    async def embed(self, texts) -> list:
        """텍스트 임베딩 - LightRAG 호환성"""
        try:
            import numpy as np
            
            # LightRAG는 단일 텍스트나 텍스트 리스트를 전달할 수 있음
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                texts = [str(texts)]
            
            # 빈 리스트 처리
            if not texts:
                logger.warning("빈 텍스트 리스트에 대한 임베딩 요청")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            results = []
            for text in texts:
                # 각 텍스트 처리
                if not text or not str(text).strip():
                    logger.warning("빈 텍스트에 대한 임베딩, 더미 벡터 사용")
                    embedding = [0.0] * self.embedding_dim
                else:
                    try:
                        response = await self.client.embeddings.create(
                            model=self.embedding_model,
                            input=str(text).strip()
                        )
                        
                        embedding = response.data[0].embedding
                        if not embedding or len(embedding) == 0:
                            logger.warning("OpenAI로부터 빈 임베딩 응답, 더미 벡터 사용")
                            embedding = [0.0] * self.embedding_dim
                        elif len(embedding) != self.embedding_dim:
                            logger.warning(f"임베딩 차원 불일치: 반환={len(embedding)}, 예상={self.embedding_dim}")
                            # 차원 조정
                            if len(embedding) > self.embedding_dim:
                                embedding = embedding[:self.embedding_dim]
                            else:
                                embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                    except Exception as embed_error:
                        logger.error(f"개별 텍스트 임베딩 실패: {embed_error}")
                        embedding = [0.0] * self.embedding_dim
                
                results.append(embedding)
            
            # 단일 텍스트인 경우 첫 번째 결과만 반환
            if len(texts) == 1:
                return results[0]
            
            # numpy 배열로 변환하여 차원 확인
            result_array = np.array(results)
            if result_array.shape[0] == 0:
                logger.warning("결과 배열이 비어있음, 더미 벡터 반환")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"OpenAI 임베딩 처리 중 오류: {e}")
            # 안전한 더미 벡터 반환
            if isinstance(texts, list) and len(texts) > 1:
                return [[0.0] * self.embedding_dim for _ in texts]
            else:
                return [0.0] * self.embedding_dim

    @property
    def embedding_dim(self) -> int:
        """OpenAI 모델의 임베딩 차원을 반환합니다."""
        if settings.embedding_dim:
            return settings.embedding_dim
        for model_key, dim in OPENAI_EMBEDDING_DIMS.items():
            if model_key in self.embedding_model:
                logger.debug(f"{self.embedding_model} 모델의 임베딩 차원을 {dim}(으)로 감지했습니다.")
                return dim
        logger.warning(f"OpenAI 임베딩 모델({self.embedding_model})의 차원을 알 수 없습니다. .env에 EMBEDDING_DIM=값 을 설정해주세요. 기본값 1536을 사용합니다.")
        return 1536


class OpenRouterService(LLMService):
    """OpenRouter 서비스"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url
        )
        self.model = settings.openrouter_model
        # OpenRouter 임베딩을 위해 내부적으로 OpenAI 서비스 사용
        self._openai_service = OpenAIService() if settings.openai_api_key else None

    @classmethod
    async def create(cls) -> "OpenRouterService":
        return cls()

    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', settings.temperature),
                max_tokens=kwargs.get('max_tokens', settings.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter 생성 오류: {e}")
            raise
    
    async def embed(self, texts) -> list:
        """텍스트 임베딩 - OpenRouter는 임베딩을 지원하지 않으므로 OpenAI 사용"""
        try:
            import numpy as np
            
            # LightRAG는 단일 텍스트나 텍스트 리스트를 전달할 수 있음
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                texts = [str(texts)]
            
            # 빈 리스트 처리
            if not texts:
                logger.warning("빈 텍스트 리스트에 대한 임베딩 요청")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            if self._openai_service:
                return await self._openai_service.embed(texts)
            else:
                # OpenAI 서비스가 없는 경우 더미 임베딩 생성
                logger.warning("임베딩을 위해 OpenAI API 키가 필요합니다. 더미 임베딩을 사용합니다.")
                results = []
                for text in texts:
                    if not text or not str(text).strip():
                        embedding = [0.0] * self.embedding_dim
                    else:
                        import hashlib
                        hash_obj = hashlib.sha256(str(text).encode())
                        hash_bytes = hash_obj.digest()
                        embedding = [float(b) / 255.0 for b in hash_bytes[:self.embedding_dim]]
                        # 차원이 부족한 경우 0으로 채움
                        if len(embedding) < self.embedding_dim:
                            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                    results.append(embedding)
                
                # 단일 텍스트인 경우 첫 번째 결과만 반환
                if len(texts) == 1:
                    return results[0]
                return results
        except Exception as e:
            logger.error(f"OpenRouter 임베딩 처리 중 오류: {e}")
            # 안전한 더미 벡터 반환
            if isinstance(texts, list) and len(texts) > 1:
                return [[0.0] * self.embedding_dim for _ in texts]
            else:
                return [0.0] * self.embedding_dim

    @property
    def embedding_dim(self) -> int:
        """OpenRouter를 통해 사용하는 임베딩 모델의 차원을 반환합니다."""
        if self._openai_service:
            return self._openai_service.embedding_dim
        else:
            # 사용하는 더미 임베딩 차원
            return 384


class LocalLLMService(LLMService):
    """로컬 LLM API 서비스"""
    
    def __init__(self):
        self.base_url = settings.local_api_host
        self.embedding_model = settings.lightrag_embedding_model
    
    @classmethod
    async def create(cls) -> "LocalLLMService":
        """비동기 로컬 LLM 서비스 생성자"""
        instance = cls()
        # 연결 테스트
        if not await instance._test_connection():
            raise Exception(f"로컬 LLM API 연결 실패: {instance.base_url}")
        return instance
    
    async def _test_connection(self) -> bool:
        """로컬 LLM API 연결 테스트"""
        try:
            async with aiohttp.ClientSession() as session:
                # 헬스체크 엔드포인트 시도
                endpoints = ["/health", "/v1/models", "/ping", "/"]
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}", timeout=5) as response:
                            if response.status in [200, 404]:  # 404도 서버가 응답하는 것으로 간주
                                logger.info(f"로컬 LLM API 서버 연결 확인: {self.base_url}")
                                return True
                    except:
                        continue
                return False
        except Exception as e:
            logger.error(f"로컬 LLM API 연결 테스트 실패: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        try:
            # OpenAI 호환 API 형식으로 요청
            payload = {
                "model": kwargs.get("model", "default"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', settings.temperature),
                "max_tokens": kwargs.get('max_tokens', settings.max_tokens),
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"로컬 LLM API 생성 실패 (HTTP {response.status}): {error_text}")
                        raise Exception(f"로컬 LLM API 요청 실패: {response.status}")
        except Exception as e:
            logger.error(f"로컬 LLM 생성 오류: {e}")
            raise
    
    async def embed(self, texts) -> list:
        """텍스트 임베딩 - 로컬 API 또는 Ollama 폴백"""
        try:
            import numpy as np
            
            # 입력 처리
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                texts = [str(texts)]
            
            if not texts:
                logger.warning("빈 텍스트 리스트에 대한 임베딩 요청")
                return np.zeros((1, self.embedding_dim)).tolist()
            
            results = []
            for text in texts:
                if not text or not str(text).strip():
                    embedding = [0.0] * self.embedding_dim
                else:
                    try:
                        # 로컬 임베딩 API 시도
                        payload = {
                            "model": self.embedding_model,
                            "input": str(text).strip()
                        }
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{self.base_url}/v1/embeddings",
                                json=payload,
                                headers={"Content-Type": "application/json"}
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    embedding = result["data"][0]["embedding"]
                                else:
                                    # 로컬 임베딩 실패시 Ollama 폴백
                                    logger.warning(f"로컬 임베딩 실패, Ollama 폴백 사용")
                                    embedding = await self._fallback_embedding(text)
                    except Exception as embed_error:
                        logger.warning(f"로컬 임베딩 실패: {embed_error}, Ollama 폴백 사용")
                        embedding = await self._fallback_embedding(text)
                
                results.append(embedding)
            
            if len(texts) == 1:
                return results[0]
            return results
            
        except Exception as e:
            logger.error(f"로컬 LLM 임베딩 처리 중 오류: {e}")
            # 더미 벡터 반환
            dummy_dim = self.embedding_dim
            if isinstance(texts, list) and len(texts) > 1:
                return [[0.0] * dummy_dim for _ in texts]
            else:
                return [0.0] * dummy_dim
    
    async def _fallback_embedding(self, text: str) -> list:
        """Ollama 폴백 임베딩"""
        try:
            # Ollama 클라이언트 생성
            client = ollama.AsyncClient(host=settings.ollama_host)
            response = await client.embeddings(
                model=settings.ollama_embedding_model,
                prompt=str(text).strip()
            )
            embedding = response.get('embedding', [])
            if not embedding:
                return [0.0] * self.embedding_dim
            return embedding
        except Exception as e:
            logger.error(f"Ollama 폴백 임베딩 실패: {e}")
            return [0.0] * self.embedding_dim
    
    @property
    def embedding_dim(self) -> int:
        """로컬 LLM 임베딩 차원"""
        if settings.embedding_dim:
            return settings.embedding_dim
        # OpenAI 호환 모델 기본값
        for model_key, dim in OPENAI_EMBEDDING_DIMS.items():
            if model_key in self.embedding_model:
                return dim
        # 기본값
        return 1536


async def get_llm_service() -> LLMService:
    """설정에 따라 적절한 LLM 서비스 반환 (비동기)"""
    provider = settings.llm_provider.lower()

    if provider == "auto":
        provider = settings.get_llm_service()

    if provider == "local":
        logger.info("로컬 LLM API 서비스 사용 (LLM_PROVIDER)")
        try:
            return await LocalLLMService.create()
        except Exception as e:
            logger.warning(f"로컬 LLM API 서비스 실패, Ollama로 폴백: {e}")
            return await OllamaService.create()
    elif provider == "openrouter":
        logger.info("OpenRouter 서비스 사용 (LLM_PROVIDER)")
        return await OpenRouterService.create()
    elif provider == "openai":
        logger.info("OpenAI 서비스 사용 (LLM_PROVIDER)")
        return await OpenAIService.create()
    elif provider == "ollama":
        logger.info("Ollama 서비스 사용 (LLM_PROVIDER)")
        return await OllamaService.create()
    else:
        logger.warning(f"LLM_PROVIDER 값 '{provider}'를 인식할 수 없습니다. 기본값 ollama 사용")
        return await OllamaService.create() 