"""LightRAG 기반 RAG 서비스"""
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
from lightrag import LightRAG
from src.Config.config import settings
from src.Service.llm_service import get_llm_service, LLMService
from src.Service.document_loader import DocumentLoader

# 전역 LLM 서비스 (직렬화 문제 해결을 위해)
_global_llm_service: Optional[LLMService] = None
# 동시성 제어를 위한 세마포어 (최대 2개의 동시 요청만 허용)
_llm_semaphore = asyncio.Semaphore(2)
_embedding_semaphore = asyncio.Semaphore(3)


class RAGService:
    """LightRAG를 활용한 RAG 서비스"""

    # __init__은 비동기일 수 없으므로, 비동기 초기화를 위한 별도 메서드를 만듭니다.
    def __init__(self, llm_service: LLMService, rag_instance: Optional[LightRAG]):
        self.document_loader = DocumentLoader()
        self.llm_service = llm_service
        self.rag = rag_instance

    @classmethod
    async def create(cls) -> "RAGService":
        """RAGService의 비동기 생성자"""
        llm_service = await get_llm_service()
        rag_instance = await cls.a_initialize_rag(llm_service)
        return cls(llm_service, rag_instance)

    @staticmethod
    async def _check_llm_health(llm_service: LLMService, max_attempts: int = 3) -> bool:
        """LLM 서비스 상태 확인"""
        for attempt in range(max_attempts):
            try:
                # 간단한 테스트 프롬프트로 서비스 상태 확인
                async with _llm_semaphore:  # 세마포어 사용
                    test_response = await llm_service.generate(
                        "Hello", 
                        temperature=0.1, 
                        max_tokens=10
                    )
                if test_response and test_response.strip():
                    logger.info(f"LLM 서비스 상태 확인 완료 (시도 {attempt + 1}/{max_attempts})")
                    return True
                else:
                    logger.warning(f"LLM 서비스 빈 응답 (시도 {attempt + 1}/{max_attempts})")
            except Exception as e:
                logger.warning(f"LLM 서비스 상태 확인 실패 (시도 {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # 지수적 백오프 (2, 4, 8초)
                    logger.info(f"LLM 서비스 안정화 대기 중... ({wait_time}초)")
                    await asyncio.sleep(wait_time)
        
        logger.error("LLM 서비스 상태 확인 실패")
        return False

    @staticmethod
    async def a_initialize_rag(llm_service: LLMService) -> Optional[LightRAG]:
        """LightRAG 초기화 (비동기)"""
        try:
            settings.create_directories()

            # LLM 서비스 상태 확인
            logger.info("LLM 서비스 상태 확인 중...")
            if not await RAGService._check_llm_health(llm_service):
                logger.warning("LLM 서비스가 불안정하지만 계속 진행합니다...")

            # 전역 변수로 LLM 서비스 저장 (직렬화 문제 해결)
            global _global_llm_service
            _global_llm_service = llm_service

            # 직렬화 가능한 래퍼 함수들 생성
            async def llm_model_func(prompt: str, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
                # LightRAG에서 전달하는 매개변수들을 처리하고 LLM 서비스가 지원하는 것만 전달
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k in ['temperature', 'max_tokens']
                }
                
                # 시스템 프롬프트가 있는 경우 포함
                final_prompt = prompt
                if system_prompt:
                    final_prompt = f"{system_prompt}\n\n{prompt}"
                
                # 세마포어를 사용한 동시성 제어
                async with _llm_semaphore:
                    # 재시도 로직 (최대 3회, 개선된 백오프)
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            if _global_llm_service is None:
                                logger.error("전역 LLM 서비스가 없습니다")
                                return "LLM 서비스가 초기화되지 않았습니다."
                            
                            result = await _global_llm_service.generate(final_prompt, **filtered_kwargs)
                            if result and result.strip():  # 비어있지 않은 응답만 반환
                                logger.debug(f"LLM 응답 생성 성공 (길이: {len(result)})")
                                return result
                            else:
                                logger.warning(f"LLM이 빈 응답을 반환했습니다 (시도 {attempt + 1}/{max_retries})")
                        except Exception as e:
                            logger.error(f"LLM 생성 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                            if attempt == max_retries - 1:  # 마지막 시도
                                logger.error("모든 재시도 실패, 기본 응답 반환")
                                return f"LLM 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
                            
                            # 재시도 전 대기 시간 증가 (지수적 백오프)
                            wait_time = (2 ** attempt) + 2  # 4, 6, 10초로 증가
                            logger.info(f"LLM 재시도 대기 중... ({wait_time}초)")
                            await asyncio.sleep(wait_time)
                    
                    return "LLM 서비스 응답을 받을 수 없습니다."

            async def embedding_func(texts):
                """LightRAG 호환 임베딩 함수 (동시성 제어 포함)"""
                async with _embedding_semaphore:  # 임베딩 세마포어 사용
                    try:
                        if _global_llm_service is None:
                            logger.error("전역 LLM 서비스가 없습니다")
                            dummy_dim = 1024  # 기본 차원
                            if isinstance(texts, list) and len(texts) > 1:
                                return [[0.0] * dummy_dim for _ in texts]
                            else:
                                return [0.0] * dummy_dim
                        
                        logger.debug(f"임베딩 요청: {type(texts)}, 길이: {len(texts) if isinstance(texts, list) else 'N/A'}")
                        
                        # 입력 처리 - LightRAG는 다양한 형태로 텍스트를 전달할 수 있음
                        if isinstance(texts, str):
                            # 단일 문자열인 경우
                            if not texts.strip():
                                logger.warning("빈 문자열에 대한 임베딩")
                                return [0.0] * _global_llm_service.embedding_dim
                            
                            # 재시도 로직 추가
                            max_retries = 2
                            for attempt in range(max_retries):
                                try:
                                    result = await _global_llm_service.embed(texts)
                                    logger.debug(f"임베딩 결과 차원: {len(result)}")
                                    return result  # 리스트 형태로 반환
                                except Exception as e:
                                    logger.warning(f"임베딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(1)  # 1초 대기
                                    else:
                                        raise
                                
                        elif isinstance(texts, list):
                            # 리스트인 경우
                            if not texts:
                                logger.warning("빈 텍스트 리스트에 대한 임베딩")
                                return [[0.0] * _global_llm_service.embedding_dim]
                            
                            # 각 텍스트에 대해 임베딩 생성 (순차 처리로 안정성 향상)
                            results = []
                            for i, text in enumerate(texts):
                                if not text or not str(text).strip():
                                    logger.warning(f"빈 텍스트 항목 {i}에 대한 임베딩")
                                    embedding = [0.0] * _global_llm_service.embedding_dim
                                else:
                                    # 재시도 로직 추가
                                    max_retries = 2
                                    embedding = None
                                    for attempt in range(max_retries):
                                        try:
                                            embedding = await _global_llm_service.embed(str(text))
                                            logger.debug(f"텍스트 {i} 임베딩 차원: {len(embedding)}")
                                            break
                                        except Exception as e:
                                            logger.warning(f"텍스트 {i} 임베딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                            if attempt < max_retries - 1:
                                                await asyncio.sleep(0.5)  # 0.5초 대기
                                            else:
                                                logger.error(f"텍스트 {i} 임베딩 완전 실패, 더미 벡터 사용")
                                                embedding = [0.0] * _global_llm_service.embedding_dim
                                results.append(embedding)
                                
                                # 텍스트 간 약간의 지연으로 서버 부하 완화
                                if i < len(texts) - 1:
                                    await asyncio.sleep(0.1)
                            
                            return results  # 중첩 리스트 형태로 반환
                        else:
                            # 기타 형태 처리
                            text_str = str(texts)
                            if not text_str.strip():
                                return [0.0] * _global_llm_service.embedding_dim
                            
                            # 재시도 로직 추가
                            max_retries = 2
                            for attempt in range(max_retries):
                                try:
                                    result = await _global_llm_service.embed(text_str)
                                    return result
                                except Exception as e:
                                    logger.warning(f"임베딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(1)
                                    else:
                                        raise
                                
                    except Exception as e:
                        logger.error(f"임베딩 함수에서 오류 발생: {e}")
                        # 더미 벡터 반환
                        dummy_dim = _global_llm_service.embedding_dim if _global_llm_service else 1024
                        if isinstance(texts, list) and len(texts) > 1:
                            # 리스트인 경우 각 항목에 대해 더미 벡터 생성
                            return [[0.0] * dummy_dim for _ in texts]
                        else:
                            # 단일 벡터 반환
                            return [0.0] * dummy_dim

            # LightRAG에서 EmbeddingFunc 래퍼 사용
            from lightrag.utils import EmbeddingFunc
            
            embedding_wrapper = EmbeddingFunc(
                embedding_dim=llm_service.embedding_dim,
                max_token_size=8192,
                func=embedding_func
            )
            
            rag_instance = LightRAG(
                working_dir=str(settings.lightrag_working_dir),
                llm_model_func=llm_model_func,
                embedding_func=embedding_wrapper,
                chunk_token_size=settings.lightrag_chunk_size,
                chunk_overlap_token_size=settings.lightrag_chunk_overlap,
            )
            
            logger.debug("LightRAG 저장소 초기화 시작...")
            # LightRAG 저장소 초기화 (필수)
            await rag_instance.initialize_storages()
            logger.debug("LightRAG 저장소 초기화 완료")
            
            logger.debug("파이프라인 상태 초기화 시작...")
            # 파이프라인 상태 초기화 (필수)
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
            logger.debug("파이프라인 상태 초기화 완료")
            
            provider = settings.llm_provider
            if provider == 'auto':
                provider = settings.get_llm_service()  # 실제 사용될 서비스
            logger.info(f"LightRAG 초기화 완료 ({provider} 서비스, 임베딩 차원: {embedding_wrapper.embedding_dim})")
            logger.debug(f"작업 디렉토리: {settings.lightrag_working_dir}")
            logger.debug(f"청크 크기: {settings.lightrag_chunk_size}, 오버랩: {settings.lightrag_chunk_overlap}")
            logger.info("동시성 제어: LLM 세마포어=2, 임베딩 세마포어=3")
            return rag_instance
        except Exception as e:
            logger.error(f"LightRAG 초기화 실패: {e}")
            raise

    async def insert_documents(self, documents: Optional[List[Dict[str, str]]] = None):
        """문서를 RAG에 삽입"""
        if self.rag is None:
            logger.error("RAG 서비스가 초기화되지 않았습니다.")
            return
        try:
            if documents is None:
                documents = self.document_loader.load_documents()

            if not documents:
                logger.warning("삽입할 문서가 없습니다.")
                return

            # LightRAG 전용 file_paths 매개변수 사용
            contents = [doc['content'] for doc in documents]
            file_paths = [doc.get('relative_path', doc['name']) for doc in documents]
            
            # 문서 내용 로그 (디버깅용)
            for i, doc in enumerate(documents):
                logger.debug(f"문서 {i+1} ({doc['path']}) 내용 (첫 200자): {doc['content'][:200]}")

            logger.info(f"{len(documents)}개 문서 삽입 시작...")
            # LightRAG의 file_paths 매개변수를 사용하여 올바른 소스 정보 제공
            try:
                await self.rag.ainsert(contents, file_paths=file_paths)
                logger.info(f"문서 삽입 완료 - 총 {len(contents)}개 문서, 파일 경로 정보 포함")
            except Exception as e:
                logger.warning(f"file_paths 방식 실패: {e}, 기본 방식 시도...")
                # 기본 방식으로 fallback
                await self.rag.ainsert(contents)
                logger.info(f"문서 삽입 완료 - 총 {len(contents)}개 문서 (기본 방식)")
            
            # 인덱싱 상태 확인
            try:
                storage_info = await self.get_indexed_info()
                logger.info(f"인덱스 상태: {storage_info}")
            except Exception as e:
                logger.warning(f"인덱스 상태 확인 실패: {e}")

        except Exception as e:
            logger.error(f"문서 삽입 실패: {e}")
            raise

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """RAG에 질의"""
        if self.rag is None:
            logger.error("RAG 서비스가 초기화되지 않았습니다.")
            return "RAG 서비스가 초기화되지 않았습니다."
        try:
            logger.debug(f"질의: {question} (모드: {mode})")

            # 한국어 프롬프트 시스템을 LLM 단계에서 처리하도록 변경
            original_question = question

            # LightRAG API 호환성 수정
            from lightrag import QueryParam
            param = QueryParam(mode=mode)
            
            logger.debug("LightRAG 질의 시작...")
            response = await self.rag.aquery(original_question, param=param)
            logger.debug(f"LightRAG 응답 타입: {type(response)}")
            logger.debug(f"LightRAG 응답 내용 (첫 200자): {str(response)[:200]}")
            
            # 응답이 dict 형태인 경우 처리
            if isinstance(response, dict):
                if 'response' in response:
                    answer = response['response']
                elif 'answer' in response:
                    answer = response['answer']
                else:
                    # dict의 첫 번째 값을 반환하거나 문자열로 변환
                    answer = str(response)
            else:
                answer = response
            
            logger.debug(f"처리된 답변: {answer}")
            
            # no-context 응답인 경우 다른 모드로 재시도
            if answer and "[no-context]" in str(answer):
                logger.warning(f"{mode} 모드에서 컨텍스트를 찾지 못했습니다. naive 모드로 재시도...")
                try:
                    param = QueryParam(mode="naive")
                    response = await self.rag.aquery(original_question, param=param)
                    if isinstance(response, dict):
                        if 'response' in response:
                            answer = response['response']
                        elif 'answer' in response:
                            answer = response['answer']
                        else:
                            answer = str(response)
                    else:
                        answer = response
                    logger.debug(f"naive 모드 응답: {answer}")
                except Exception as e:
                    logger.warning(f"naive 모드 재시도 실패: {e}")
                
            # 한국어 질문에 대해서는 LLM에게 한국어로 답변하도록 후처리
            if answer and self._is_korean(original_question) and not self._is_korean(answer) and "[no-context]" not in str(answer):
                # 한국어로 다시 답변 요청
                korean_prompt = f"다음 답변을 한국어로 번역해주세요:\n\n{answer}"
                try:
                    if _global_llm_service is not None:
                        async with _llm_semaphore:  # 세마포어 사용
                            korean_answer = await _global_llm_service.generate(korean_prompt)
                            return korean_answer if korean_answer else answer
                    else:
                        logger.warning("전역 LLM 서비스가 없어 한국어 번역을 건너뜁니다")
                        return answer
                except Exception as e:
                    logger.warning(f"한국어 번역 실패: {e}")
                    return answer
            
            return answer if answer else "답변을 생성할 수 없습니다."

        except Exception as e:
            logger.error(f"질의 실패: {e}")
            logger.exception("질의 실패 상세 정보")
            return f"오류가 발생했습니다: {str(e)}"

    def _is_korean(self, text: str) -> bool:
        """텍스트가 한국어인지 확인"""
        korean_chars = sum(1 for char in text if '가' <= char <= '힣')
        return korean_chars / len(text) > 0.3 if text else False

    async def get_indexed_info(self) -> Dict[str, Any]:
        """인덱싱된 정보 반환"""
        try:
            storage_path = Path(settings.lightrag_working_dir)
            info: Dict[str, Any] = {
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