"""LLM 서비스 관리 모듈"""
import os
import aiohttp
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import ollama
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from src.Config.config import settings
import asyncio

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
                logger.info(f"Ollama에서 다음 모델을 성공적으로 인식했습니다: {', '.join([m for m in installed_models if m])}")
                
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
        """텍스트 생성 - 안정성 개선 및 올바른 API 호출"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Ollama 클라이언트 옵션 설정
                options = {
                    'temperature': kwargs.get('temperature', settings.temperature),
                    'num_predict': kwargs.get('max_tokens', settings.max_tokens)  # max_tokens → num_predict
                }
                
                # 올바른 Ollama API 호출 방식 사용
                response = await self.client.chat(
                    model=self.chat_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options=options
                )
                
                # 응답 내용 추출
                if response and 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    if content and content.strip():
                        logger.debug(f"Ollama 응답 생성 성공 (시도 {attempt + 1}/{max_retries}, 길이: {len(content)})")
                        return content
                    else:
                        logger.warning(f"Ollama가 빈 응답을 반환했습니다 (시도 {attempt + 1}/{max_retries})")
                else:
                    logger.warning(f"Ollama 응답 형식이 예상과 다릅니다 (시도 {attempt + 1}/{max_retries}): {response}")
                
                # 빈 응답인 경우 재시도 (마지막 시도가 아닌 경우)
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))  # 지수적 백오프
                    continue
                else:
                    return "Ollama에서 응답을 생성할 수 없습니다."
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Ollama 생성 오류 (시도 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 재시도 가능한 오류인지 확인
                if attempt < max_retries - 1:
                    # 연결 오류나 일시적 오류의 경우 재시도
                    if any(err_type in error_msg.lower() for err_type in ['eof', 'connection', 'timeout', '500']):
                        wait_time = base_delay * (2 ** attempt)
                        logger.info(f"Ollama 재시도 대기 중... ({wait_time}초)")
                        await asyncio.sleep(wait_time)
                        continue
                
                # 마지막 시도이거나 재시도 불가능한 오류
                if attempt == max_retries - 1:
                    logger.error(f"Ollama 모든 재시도 실패: {error_msg}")
                    raise
                else:
                    # 즉시 실패해야 하는 오류 (API 키 문제 등)
                    raise
        
        # 모든 재시도 실패 시 기본 응답 반환
        return "Ollama 서비스에 일시적인 문제가 발생했습니다."
    
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
                        
                        embedding = response.data[0].embedding if response.data[0].embedding else []
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
        self._lock = asyncio.Lock()  # 동시 요청을 제어하기 위한 Lock
    
    @classmethod
    async def create(cls) -> "LocalLLMService":
        """비동기 로컬 LLM 서비스 생성자"""
        instance = cls()
        # 연결 테스트
        if not await instance._test_connection():
            raise Exception(f"로컬 LLM API 연결 실패: {instance.base_url}")
        return instance
    
    async def _test_connection(self) -> bool:
        """로컬 LLM API 연결 테스트 - 세션 상태 오류 고려"""
        try:
            async with aiohttp.ClientSession() as session:
                # 먼저 기본 경로로 서버 상태 확인 (OPTIONS 요청)
                try:
                    async with session.options(self.base_url, timeout=3) as response:
                        if response.status in [200, 405]:  # OPTIONS를 지원하지 않더라도 서버가 응답하면 OK
                            logger.info(f"로컬 LLM API 서버 연결 확인: {self.base_url} (상태: {response.status})")
                            return True
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
                    pass
                
                # OPTIONS가 실패하면 실제 채팅 엔드포인트로 테스트
                url = f"{self.base_url}{settings.local_api_chat_endpoint}"
                payload = {"content": "connection_test", "type": "user"}
                
                try:
                    async with session.post(url, json=payload, timeout=5) as response:
                        # 세션 상태 오류도 연결 성공으로 간주
                        if response.status == 200:
                            logger.info(f"로컬 LLM API 서버 연결 확인: {url} (상태: 200)")
                            return True
                        elif response.status == 500:
                            # 500 오류의 세부 내용 확인
                            try:
                                error_data = await response.json()
                                error_message = str(error_data)
                                # 세션 상태 관련 오류는 실제로는 연결 성공을 의미
                                if "waiting for user input" in error_message.lower() or "session" in error_message.lower():
                                    logger.info(f"로컬 LLM API 서버 연결 확인: {url} (세션 상태 오류, 연결 성공)")
                                    return True
                                else:
                                    logger.warning(f"로컬 LLM API 서버 오류: {error_message}")
                                    return False
                            except:
                                logger.warning(f"로컬 LLM API 서버 HTTP 500 오류")
                                return False
                        else:
                            logger.warning(f"로컬 LLM API 서버 응답: {response.status}")
                            return response.status < 500  # 4xx는 연결됨, 5xx는 서버 오류
                            
                except aiohttp.ClientConnectorError:
                    logger.error(f"로컬 LLM API 서버에 연결할 수 없습니다: {url}")
                    return False
                except asyncio.TimeoutError:
                    logger.error(f"로컬 LLM API 서버 응답 시간 초과: {url}")
                    return False
                    
        except Exception as e:
            logger.error(f"로컬 LLM API 연결 테스트 중 예외 발생: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성 - 세션 상태 오류 처리 개선 및 Lock으로 동시 요청 제어"""
        async with self._lock: # 한 번에 하나의 요청만 처리하도록 Lock 사용
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 제공된 curl 명령어 형식으로 요청
                    payload = {
                        "content": prompt,
                        "type": "user",
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}{settings.local_api_chat_endpoint}",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=30  # 타임아웃 증가
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                # 응답에서 content 필드를 찾아서 반환
                                if "content" in result and result["content"]:
                                    return result["content"]
                                # 다른 가능한 필드도 확인 (예: message)
                                elif "message" in result and isinstance(result["message"], dict) and "content" in result["message"]:
                                    return result["message"]["content"]
                                else:
                                    logger.warning(f"로컬 LLM 응답에서 'content' 필드를 찾을 수 없거나 비어 있습니다. 빈 문자열을 반환합니다. 전체 응답: {result}")
                                    return ""
                            elif response.status == 500:
                                error_text = await response.text()
                                try:
                                    error_data = await response.json() if error_text else {}
                                    error_message = str(error_data)
                                    
                                    # 세션 상태 오류 처리
                                    if "waiting for user input" in error_message.lower():
                                        if attempt < max_retries - 1:
                                            logger.warning(f"세션 상태 오류 감지, 세션 초기화 시도 {attempt + 1}/{max_retries}")
                                            await self._reset_session()
                                            await asyncio.sleep(1)  # 1초 대기 후 재시도
                                            continue
                                        else:
                                            logger.error("세션 상태 오류: 모든 재시도 실패")
                                            raise Exception("로컬 LLM API 세션 상태 오류")
                                    else:
                                        logger.error(f"로컬 LLM API 생성 실패 (HTTP 500): {error_text}")
                                        raise Exception(f"로컬 LLM API 요청 실패: 500")
                                except Exception as parse_error:
                                    logger.error(f"로컬 LLM API 오류 응답 파싱 실패: {parse_error}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(2)
                                        continue
                                    raise Exception(f"로컬 LLM API 요청 실패: {response.status}")
                            else:
                                error_text = await response.text()
                                logger.error(f"로컬 LLM API 생성 실패 (HTTP {response.status}): {error_text}")
                                raise Exception(f"로컬 LLM API 요청 실패: {response.status}")
                                
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 지수적 백오프
                        logger.warning(f"로컬 LLM 생성 오류 (시도 {attempt + 1}/{max_retries}): {e}, {wait_time}초 후 재시도")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"로컬 LLM 생성 오류: {e}")
                        raise
        return "" # 모든 재시도 실패 시 빈 문자열 반환

    async def _reset_session(self) -> bool:
        """세션 초기화 시도"""
        try:
            # 세션 리셋을 위한 특별한 엔드포인트가 있다면 사용
            reset_endpoints = ["/reset", "/session/reset", "/api/reset"]
            
            async with aiohttp.ClientSession() as session:
                for endpoint in reset_endpoints:
                    try:
                        async with session.post(f"{self.base_url}{endpoint}", timeout=5) as response:
                            if response.status == 200:
                                logger.info(f"세션 초기화 성공: {endpoint}")
                                return True
                    except:
                        continue
                        
                # 리셋 엔드포인트가 없다면 기본 상태 확인
                try:
                    async with session.get(f"{self.base_url}/health", timeout=3) as response:
                        if response.status == 200:
                            logger.info("서버 상태 확인 완료")
                            return True
                except:
                    pass
                    
            logger.warning("세션 초기화 실패")
            return False
            
        except Exception as e:
            logger.error(f"세션 초기화 중 오류: {e}")
            return False
    
    async def embed(self, texts) -> list:
        """텍스트 임베딩 - Ollama 강제 사용"""
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
                    # 로컬 임베딩을 시도하지 않고 항상 Ollama 폴백 사용
                    logger.debug("로컬 LLM 공급자 모드: 임베딩을 위해 Ollama를 사용합니다.")
                    embedding = await self._fallback_embedding(text)
                
                results.append(embedding)
            
            if len(texts) == 1:
                return results[0]
            return results
            
        except Exception as e:
            logger.error(f"로컬 LLM 임베딩 처리 중(Ollama 폴백 사용): {e}")
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
        """임베딩 차원을 Ollama에서 가져옵니다."""
        if settings.embedding_dim:
            return settings.embedding_dim
        
        embedding_model = settings.ollama_embedding_model
        for model_key, dim in OLLAMA_EMBEDDING_DIMS.items():
            if model_key in embedding_model:
                logger.debug(f"Ollama 임베딩 모델({embedding_model})의 차원을 {dim}(으)로 감지했습니다.")
                return dim
        logger.warning(f"Ollama 임베딩 모델({embedding_model})의 차원을 알 수 없습니다. .env에 EMBEDDING_DIM=값 을 설정해주세요. 기본값 768을 사용합니다.")
        return 768


async def get_llm_service() -> LLMService:
    """대화용 LLM 서비스 생성 및 반환"""
    provider = settings.llm_provider
    
    if provider == "auto":
        provider = settings.get_llm_service()
    
    logger.info(f"대화용 LLM 서비스 초기화: {provider}")
    
    try:
        if provider == "local":
            return await LocalLLMService.create()
        elif provider == "ollama":
            return await OllamaService.create()
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다")
            return await OpenAIService.create()
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API 키가 설정되지 않았습니다")
            return await OpenRouterService.create()
        else:
            raise ValueError(f"지원되지 않는 LLM 제공자: {provider}")
    except Exception as e:
        logger.error(f"대화용 LLM 서비스 생성 실패 ({provider}): {e}")
        # 폴백: Ollama 서비스 시도
        if provider != "ollama":
            logger.info("Ollama 서비스로 폴백 시도...")
            try:
                return await OllamaService.create()
            except Exception as fallback_error:
                logger.error(f"Ollama 폴백도 실패: {fallback_error}")
        raise


async def get_embedding_llm_service() -> LLMService:
    """Embedding 전용 LLM 서비스 생성 및 반환 - local일 때는 항상 ollama 사용"""
    provider = settings.get_embedding_llm_service()
    
    logger.info(f"Embedding용 LLM 서비스 초기화: {provider}")
    
    try:
        if provider == "ollama":
            return await OllamaService.create()
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다")
            return await OpenAIService.create()
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API 키가 설정되지 않았습니다")
            return await OpenRouterService.create()
        else:
            raise ValueError(f"지원되지 않는 Embedding LLM 제공자: {provider}")
    except Exception as e:
        logger.error(f"Embedding용 LLM 서비스 생성 실패 ({provider}): {e}")
        # 폴백: Ollama 서비스 시도 (embedding은 항상 ollama로 폴백)
        if provider != "ollama":
            logger.info("Embedding용 Ollama 서비스로 폴백 시도...")
            try:
                return await OllamaService.create()
            except Exception as fallback_error:
                logger.error(f"Embedding용 Ollama 폴백도 실패: {fallback_error}")
        raise


async def get_kg_llm_service() -> LLMService:
    """Knowledge Graph 전용 LLM 서비스 생성 및 반환 - local일 때는 항상 ollama 사용"""
    provider = settings.get_kg_llm_service()
    
    logger.info(f"Knowledge Graph용 LLM 서비스 초기화: {provider}")
    
    try:
        if provider == "ollama":
            return await OllamaService.create()
        elif provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다")
            return await OpenAIService.create()
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API 키가 설정되지 않았습니다")
            return await OpenRouterService.create()
        elif provider == "local":
            return await LocalLLMService.create()
        else:
            raise ValueError(f"지원되지 않는 Knowledge Graph LLM 제공자: {provider}")
    except Exception as e:
        logger.error(f"Knowledge Graph용 LLM 서비스 생성 실패 ({provider}): {e}")
        # 폴백: Ollama 서비스 시도 (KG는 항상 ollama로 폴백)
        if provider != "ollama":
            logger.info("Knowledge Graph용 Ollama 서비스로 폴백 시도...")
            try:
                return await OllamaService.create()
            except Exception as fallback_error:
                logger.error(f"Knowledge Graph용 Ollama 폴백도 실패: {fallback_error}")
        raise 