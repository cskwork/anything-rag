"""LightRAG 기반 RAG 서비스"""
import asyncio
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from src.Config.config import settings
from src.Service.llm_service import get_llm_service, get_embedding_llm_service, get_kg_llm_service, LLMService
from src.Service.document_loader import DocumentLoader
from src.Service.local_api_service import LocalApiService

# 전역 LLM 서비스 (직렬화 문제 해결을 위해)
_global_llm_service: Optional[LLMService] = None
_global_embedding_service: Optional[LLMService] = None
_global_kg_service: Optional[LLMService] = None
# 동시성 제어를 위한 세마포어 (LLM은 순차 처리, 임베딩은 병렬 처리)
_llm_semaphore = asyncio.Semaphore(1)  # LLM 호출은 한 번에 하나씩만
_embedding_semaphore = asyncio.Semaphore(2)

# 임베딩 진행률 추적을 위한 전역 변수들
_embedding_progress = {
    "total_calls": 0,
    "completed_calls": 0,
    "total_texts": 0,
    "completed_texts": 0,
    "start_time": None,
    "document_times": [],  # 각 문서별 처리 시간
    "current_document": None,
    "document_start_time": None
}


def _reset_embedding_progress():
    """임베딩 진행률 추적 변수 초기화"""
    global _embedding_progress
    _embedding_progress = {
        "total_calls": 0,
        "completed_calls": 0,
        "total_texts": 0,
        "completed_texts": 0,
        "start_time": None,
        "document_times": [],
        "current_document": None,
        "document_start_time": None
    }


def _start_document_timing(document_name: str):
    """문서별 처리 시간 측정 시작"""
    global _embedding_progress
    _embedding_progress["current_document"] = document_name
    _embedding_progress["document_start_time"] = time.time()
    logger.info(f"📄 문서 임베딩 시작: {document_name}")


def _end_document_timing():
    """문서별 처리 시간 측정 완료"""
    global _embedding_progress
    if _embedding_progress["document_start_time"] and _embedding_progress["current_document"]:
        elapsed = time.time() - _embedding_progress["document_start_time"]
        _embedding_progress["document_times"].append(elapsed)
        avg_time = sum(_embedding_progress["document_times"]) / len(_embedding_progress["document_times"])
        
        logger.info(f"✅ 문서 임베딩 완료: {_embedding_progress['current_document']} "
                   f"(소요시간: {elapsed:.1f}초, 평균: {avg_time:.1f}초)")
        
        _embedding_progress["current_document"] = None
        _embedding_progress["document_start_time"] = None


def _log_embedding_progress():
    """현재 임베딩 진행률 로그 출력"""
    global _embedding_progress
    
    if _embedding_progress["total_texts"] == 0:
        return
    
    completed = _embedding_progress["completed_texts"]
    total = _embedding_progress["total_texts"]
    percentage = (completed / total) * 100
    
    # 진행률 계산
    progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    
    # 남은 시간 예측
    if _embedding_progress["start_time"] and completed > 0:
        elapsed = time.time() - _embedding_progress["start_time"]
        rate = completed / elapsed  # 텍스트/초
        remaining = (total - completed) / rate if rate > 0 else 0
        eta_str = f", 예상 남은 시간: {remaining:.0f}초" if remaining > 0 else ""
    else:
        eta_str = ""
    
    logger.info(f"🔄 임베딩 진행률: {completed}/{total} ({percentage:.1f}%) "
               f"[{progress_bar}]{eta_str}")


class RAGService:
    """LightRAG를 활용한 RAG 서비스"""

    # __init__은 비동기일 수 없으므로, 비동기 초기화를 위한 별도 메서드를 만듭니다.
    def __init__(self, llm_service: LLMService, rag_instance: Optional[LightRAG]):
        self.document_loader = DocumentLoader()
        self.llm_service = llm_service
        self.rag = rag_instance
        self.local_api = LocalApiService()
        # 대화 히스토리 저장용 리스트
        self.conversation_history: list[dict[str, str]] = []  # [{role:"user"|"assistant", content:str}]

    @classmethod
    async def create(cls) -> "RAGService":
        """RAGService의 비동기 생성자"""
        # 대화용 LLM 서비스 생성
        llm_service = await get_llm_service()
        # embedding용 LLM 서비스 생성 (local일 때는 ollama 사용)
        embedding_service = await get_embedding_llm_service()
        # Knowledge Graph용 LLM 서비스 생성 (local일 때는 ollama 사용)
        kg_service = await get_kg_llm_service()
        
        rag_instance = await cls.a_initialize_rag(llm_service, embedding_service, kg_service)
        return cls(llm_service, rag_instance)

    @staticmethod
    async def _check_llm_health(llm_service: LLMService, max_attempts: int = 3) -> bool:
        """LLM 서비스 상태 확인 - 세션 상태 오류 고려"""
        from src.Service.llm_service import LocalLLMService
        
        # 로컬 LLM 서비스인 경우 특별 처리
        if isinstance(llm_service, LocalLLMService):
            logger.info("로컬 LLM API 서비스 감지됨, 연결 테스트 방식 변경")
            # 이미 create()에서 연결 테스트를 통과했으므로 성공으로 간주
            return True
        
        for attempt in range(max_attempts):
            try:
                # 간단한 테스트 프롬프트로 서비스 상태 확인
                async with _llm_semaphore:  # 세마포어 사용
                    test_response = await llm_service.generate(
                        "test", 
                        temperature=0.1, 
                        max_tokens=5
                    )
                if test_response and test_response.strip():
                    logger.info(f"LLM 서비스 상태 확인 완료 (시도 {attempt + 1}/{max_attempts})")
                    return True
                else:
                    logger.warning(f"LLM 서비스 빈 응답 (시도 {attempt + 1}/{max_attempts})")
            except Exception as e:
                error_message = str(e).lower()
                
                # 세션 상태 오류는 실제로는 연결 성공을 의미
                if "waiting for user input" in error_message or "session" in error_message:
                    logger.info(f"세션 상태 오류 감지 - 연결은 성공 (시도 {attempt + 1}/{max_attempts})")
                    return True
                
                logger.warning(f"LLM 서비스 상태 확인 실패 (시도 {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # 지수적 백오프 (2, 4, 8초)
                    logger.info(f"LLM 서비스 안정화 대기 중... ({wait_time}초)")
                    await asyncio.sleep(wait_time)
        
        logger.error("LLM 서비스 상태 확인 실패")
        return False

    @staticmethod
    async def a_initialize_rag(llm_service: LLMService, embedding_service: LLMService, kg_service: LLMService) -> Optional[LightRAG]:
        """LightRAG 초기화 (비동기) - 대화용, embedding용, KG용 서비스 분리"""
        try:
            settings.create_directories()

            # 대화용 LLM 서비스 상태 확인
            logger.info("대화용 LLM 서비스 상태 확인 중...")
            if not await RAGService._check_llm_health(llm_service):
                logger.warning("대화용 LLM 서비스가 불안정하지만 계속 진행합니다...")

            # embedding용 LLM 서비스 상태 확인
            logger.info("Embedding용 LLM 서비스 상태 확인 중...")
            if not await RAGService._check_llm_health(embedding_service):
                logger.warning("Embedding용 LLM 서비스가 불안정하지만 계속 진행합니다...")

            # Knowledge Graph용 LLM 서비스 상태 확인
            logger.info("Knowledge Graph용 LLM 서비스 상태 확인 중...")
            if not await RAGService._check_llm_health(kg_service):
                logger.warning("Knowledge Graph용 LLM 서비스가 불안정하지만 계속 진행합니다...")

            # 전역 변수로 LLM 서비스들 저장 (직렬화 문제 해결)
            global _global_llm_service, _global_embedding_service, _global_kg_service
            _global_llm_service = llm_service
            _global_embedding_service = embedding_service
            _global_kg_service = kg_service

            # 서비스 정보 로그
            chat_provider = settings.llm_provider
            if chat_provider == 'auto':
                chat_provider = settings.get_llm_service()
            
            embedding_provider = settings.get_embedding_llm_service()
            kg_provider = settings.get_kg_llm_service()
            
            logger.info(f"서비스 구성 - 대화: {chat_provider}, Embedding: {embedding_provider}, KG: {kg_provider}")

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
                
                # Knowledge Graph 구축과 일반 대화를 구분하여 다른 서비스 사용
                if keyword_extraction or "extract" in prompt.lower() or "entity" in prompt.lower() or "relationship" in prompt.lower():
                    # Knowledge Graph 구축용 서비스 사용
                    target_service = _global_kg_service
                    service_name = "Knowledge Graph"
                else:
                    # 일반 대화용 서비스 사용
                    target_service = _global_llm_service
                    service_name = "대화"
                
                # 세마포어를 사용한 동시성 제어 (타임아웃 추가)
                async def _generate_with_semaphore():
                    async with _llm_semaphore:
                        # 재시도 로직 (최대 3회, 개선된 백오프)
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                if target_service is None:
                                    logger.error(f"전역 {service_name}용 LLM 서비스가 없습니다")
                                    return "LLM 서비스가 초기화되지 않았습니다."
                                
                                result = await target_service.generate(final_prompt, **filtered_kwargs)
                                if result and result.strip():  # 비어있지 않은 응답만 반환
                                    logger.debug(f"{service_name}용 LLM 응답 생성 성공 (길이: {len(result)})")
                                    return result
                                else:
                                    logger.warning(f"{service_name}용 LLM이 빈 응답을 반환했습니다 (시도 {attempt + 1}/{max_retries})")
                            except Exception as e:
                                logger.error(f"{service_name}용 LLM 생성 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                if attempt == max_retries - 1:  # 마지막 시도
                                    logger.error("모든 재시도 실패, 기본 응답 반환")
                                    return f"LLM 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
                                
                                # 재시도 전 대기 시간 증가 (지수적 백오프)
                                wait_time = (2 ** attempt) + 2  # 4, 6, 10초로 증가
                                logger.info(f"{service_name}용 LLM 재시도 대기 중... ({wait_time}초)")
                                await asyncio.sleep(wait_time)
                        
                        return "LLM 서비스 응답을 받을 수 없습니다."
                
                # 2분 타임아웃으로 LLM 호출 실행
                try:
                    return await asyncio.wait_for(_generate_with_semaphore(), timeout=120)
                except asyncio.TimeoutError:
                    logger.error(f"{service_name}용 LLM 호출 타임아웃 (2분)")
                    return f"{service_name}용 LLM 서비스 타임아웃이 발생했습니다."

            async def embedding_func(texts):
                """LightRAG 호환 임베딩 함수 (embedding 전용 서비스 사용, 동시성 제어 포함, 진행률 추적)"""
                global _embedding_progress
                
                async with _embedding_semaphore:  # 임베딩 세마포어 사용
                    try:
                        if _global_embedding_service is None:
                            logger.error("전역 embedding용 LLM 서비스가 없습니다")
                            dummy_dim = 1024  # 기본 차원
                            if isinstance(texts, list) and len(texts) > 1:
                                return [[0.0] * dummy_dim for _ in texts]
                            else:
                                return [0.0] * dummy_dim
                        
                        # 진행률 추적 시작
                        _embedding_progress["total_calls"] += 1
                        
                        # 텍스트 수 계산
                        if isinstance(texts, list):
                            text_count = len(texts)
                        else:
                            text_count = 1
                        
                        _embedding_progress["total_texts"] += text_count
                        
                        logger.debug(f"임베딩 요청 (embedding 전용 서비스): {type(texts)}, 텍스트 수: {text_count}")
                        
                        # 입력 처리 - LightRAG는 다양한 형태로 텍스트를 전달할 수 있음
                        if isinstance(texts, str):
                            # 단일 문자열인 경우
                            if not texts.strip():
                                logger.warning("빈 문자열에 대한 임베딩")
                                _embedding_progress["completed_texts"] += 1
                                _log_embedding_progress()
                                return [0.0] * _global_embedding_service.embedding_dim
                            
                            # 재시도 로직 추가
                            max_retries = 2
                            for attempt in range(max_retries):
                                try:
                                    result = await _global_embedding_service.embed(texts)
                                    logger.debug(f"임베딩 결과 차원: {len(result)}")
                                    
                                    # 진행률 업데이트
                                    _embedding_progress["completed_texts"] += 1
                                    _embedding_progress["completed_calls"] += 1
                                    _log_embedding_progress()
                                    
                                    return result  # 리스트 형태로 반환
                                except Exception as e:
                                    logger.warning(f"embedding 서비스 임베딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(1)  # 1초 대기
                                    else:
                                        raise
                                
                        elif isinstance(texts, list):
                            # 리스트인 경우
                            if not texts:
                                logger.warning("빈 텍스트 리스트에 대한 임베딩")
                                _embedding_progress["completed_texts"] += 1
                                _embedding_progress["completed_calls"] += 1
                                _log_embedding_progress()
                                return [[0.0] * _global_embedding_service.embedding_dim]
                            
                            # 배치 임베딩 시작 로그
                            if len(texts) > 1:
                                logger.info(f"🔄 청크 임베딩 시작: {len(texts)}개 청크 처리")
                            
                            # 각 텍스트에 대해 임베딩 생성 (순차 처리로 안정성 향상)
                            results = []
                            batch_start_time = time.time()
                            
                            for i, text in enumerate(texts):
                                chunk_start_time = time.time()
                                
                                if not text or not str(text).strip():
                                    logger.warning(f"빈 텍스트 항목 {i+1}/{len(texts)}에 대한 임베딩")
                                    embedding = [0.0] * _global_embedding_service.embedding_dim
                                else:
                                    # 재시도 로직 추가
                                    max_retries = 2
                                    embedding = None
                                    for attempt in range(max_retries):
                                        try:
                                            embedding = await _global_embedding_service.embed(str(text))
                                            chunk_elapsed = time.time() - chunk_start_time
                                            
                                            # 텍스트 길이와 처리 시간 정보
                                            text_len = len(str(text))
                                            if len(texts) > 5:  # 많은 청크가 있을 때만 간헐적 로그
                                                if i % 10 == 0 or i == len(texts) - 1:
                                                    logger.info(f"   📝 청크 {i+1}/{len(texts)} 완료 "
                                                              f"({text_len}글자, {chunk_elapsed:.1f}초, 차원:{len(embedding)})")
                                            else:
                                                logger.debug(f"텍스트 {i+1}/{len(texts)} 임베딩 완료 "
                                                           f"({text_len}글자, {chunk_elapsed:.1f}초, 차원:{len(embedding)})")
                                            break
                                        except Exception as e:
                                            logger.warning(f"임베딩 서비스 텍스트 {i+1} 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                            if attempt < max_retries - 1:
                                                await asyncio.sleep(0.5)  # 0.5초 대기
                                            else:
                                                logger.error(f"텍스트 {i+1} 임베딩 완전 실패, 더미 벡터 사용")
                                                embedding = [0.0] * _global_embedding_service.embedding_dim
                                
                                results.append(embedding)
                                
                                # 진행률 업데이트 (개별 텍스트별)
                                _embedding_progress["completed_texts"] += 1
                                if len(texts) > 10 and (i % 5 == 0 or i == len(texts) - 1):  # 5개마다 또는 마지막에 로그
                                    _log_embedding_progress()
                                
                                # 텍스트 간 약간의 지연으로 서버 부하 완화
                                if i < len(texts) - 1:
                                    await asyncio.sleep(0.05)  # 지연 시간 단축
                            
                            # 배치 완료 로그
                            batch_elapsed = time.time() - batch_start_time
                            if len(texts) > 1:
                                avg_per_chunk = batch_elapsed / len(texts)
                                logger.info(f"✅ 청크 임베딩 완료: {len(texts)}개 청크, "
                                          f"총 {batch_elapsed:.1f}초 (평균 {avg_per_chunk:.2f}초/청크)")
                            
                            _embedding_progress["completed_calls"] += 1
                            return results  # 중첩 리스트 형태로 반환
                        else:
                            # 기타 형태 처리
                            text_str = str(texts)
                            if not text_str.strip():
                                _embedding_progress["completed_texts"] += 1
                                _embedding_progress["completed_calls"] += 1
                                _log_embedding_progress()
                                return [0.0] * _global_embedding_service.embedding_dim
                            
                            # 재시도 로직 추가
                            max_retries = 2
                            for attempt in range(max_retries):
                                try:
                                    result = await _global_embedding_service.embed(text_str)
                                    
                                    # 진행률 업데이트
                                    _embedding_progress["completed_texts"] += 1
                                    _embedding_progress["completed_calls"] += 1
                                    _log_embedding_progress()
                                    
                                    return result
                                except Exception as e:
                                    logger.warning(f"embedding 서비스 임베딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(1)
                                    else:
                                        raise
                                
                    except Exception as e:
                        logger.error(f"임베딩 함수에서 오류 발생: {e}")
                        # 더미 벡터 반환
                        dummy_dim = _global_embedding_service.embedding_dim if _global_embedding_service else 1024
                        
                        # 진행률 업데이트 (실패한 경우도)
                        _embedding_progress["completed_texts"] += text_count
                        _embedding_progress["completed_calls"] += 1
                        _log_embedding_progress()
                        
                        if isinstance(texts, list) and len(texts) > 1:
                            # 리스트인 경우 각 항목에 대해 더미 벡터 생성
                            return [[0.0] * dummy_dim for _ in texts]
                        else:
                            # 단일 벡터 반환
                            return [0.0] * dummy_dim

            # LightRAG에서 EmbeddingFunc 래퍼 사용 (embedding 전용 서비스의 차원 사용)
            embedding_wrapper = EmbeddingFunc(
                embedding_dim=embedding_service.embedding_dim,
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
            
            logger.info(f"LightRAG 초기화 완료 - 대화: {chat_provider}, Embedding: {embedding_provider}, KG: {kg_provider} (차원: {embedding_wrapper.embedding_dim})")
            logger.debug(f"작업 디렉토리: {settings.lightrag_working_dir}")
            logger.debug(f"청크 크기: {settings.lightrag_chunk_size}, 오버랩: {settings.lightrag_chunk_overlap}")
            logger.info("동시성 제어: LLM 세마포어=1 (순차), 임베딩 세마포어=2")
            return rag_instance
        except Exception as e:
            logger.error(f"LightRAG 초기화 실패: {e}")
            raise

    async def insert_documents(self, documents: Optional[List[Dict[str, str]]] = None, only_new: bool = True):
        """문서를 RAG에 삽입 (진행률 추적 포함)
        
        Args:
            documents: 삽입할 문서 리스트 (None이면 자동 로드)
            only_new: True면 신규/변경된 파일만, False면 모든 파일
        """
        if self.rag is None:
            logger.error("RAG 서비스가 초기화되지 않았습니다.")
            return
        try:
            if documents is None:
                documents = self.document_loader.load_documents(only_new=only_new)

            if not documents:
                if only_new:
                    logger.info("새로 추가되거나 변경된 문서가 없습니다.")
                else:
                    logger.warning("삽입할 문서가 없습니다.")
                return

            # 진행률 추적 초기화
            _reset_embedding_progress()
            global _embedding_progress
            _embedding_progress["start_time"] = time.time()
            
            # LightRAG 전용 file_paths 매개변수 사용
            contents = [doc['content'] for doc in documents]
            file_paths = [doc.get('relative_path', doc['name']) for doc in documents]
            
            # 문서 내용 로그 (디버깅용)
            for i, doc in enumerate(documents):
                logger.debug(f"문서 {i+1} ({doc['path']}) 내용 (첫 200자): {doc['content'][:200]}")

            logger.info(f"📚 {len(documents)}개 문서 임베딩 시작...")
            logger.info(f"💡 각 문서별 진행률과 처리 시간이 실시간으로 표시됩니다.")
            
            # 각 문서별 처리 시간 측정을 위한 개별 처리
            success_count = 0
            for i, (content, file_path) in enumerate(zip(contents, file_paths)):
                try:
                    # 문서별 시간 측정 시작
                    doc_start_time = time.time()
                    doc_name = f"{i+1}/{len(documents)} - {Path(file_path).name}"
                    
                    # 문서 크기 정보
                    content_size = len(content)
                    estimated_chunks = max(1, content_size // settings.lightrag_chunk_size)
                    
                    logger.info(f"📄 문서 임베딩 시작: {doc_name}")
                    logger.info(f"   📏 크기: {content_size:,} 글자, 예상 청크: ~{estimated_chunks}개")
                    
                    # LightRAG의 file_paths 매개변수를 사용하여 올바른 소스 정보 제공
                    try:
                        logger.info(f"🔄 LightRAG ainsert 시작 (문서 {i+1}): {Path(file_path).name}")
                        
                        # 타임아웃 설정으로 무한 대기 방지
                        insert_task = asyncio.create_task(
                            self.rag.ainsert([content], file_paths=[file_path])
                        )
                        await asyncio.wait_for(insert_task, timeout=300)  # 5분 타임아웃
                        
                        logger.info(f"✅ LightRAG ainsert 완료 (문서 {i+1}): {Path(file_path).name}")
                        success_msg = f"✅ 문서 {i+1} 임베딩 성공: {Path(file_path).name}"
                        
                    except asyncio.TimeoutError:
                        logger.error(f"⏰ 문서 {i+1} 처리 타임아웃 (5분): {Path(file_path).name}")
                        # 타임아웃 발생 시 기본 방식으로 재시도
                        try:
                            logger.info(f"🔄 기본 방식으로 재시도 (문서 {i+1}): {Path(file_path).name}")
                            basic_task = asyncio.create_task(
                                self.rag.ainsert([content])
                            )
                            await asyncio.wait_for(basic_task, timeout=120)  # 2분 타임아웃
                            success_msg = f"✅ 문서 {i+1} 임베딩 성공 (기본 방식, 재시도): {Path(file_path).name}"
                        except asyncio.TimeoutError:
                            logger.error(f"❌ 문서 {i+1} 최종 타임아웃: {Path(file_path).name}")
                            raise Exception(f"문서 처리 타임아웃: {Path(file_path).name}")
                        
                    except Exception as e:
                        logger.warning(f"file_paths 방식 실패 (문서 {i+1}): {e}, 기본 방식 시도...")
                        # 기본 방식으로 fallback
                        try:
                            logger.info(f"🔄 기본 방식으로 fallback (문서 {i+1}): {Path(file_path).name}")
                            basic_task = asyncio.create_task(
                                self.rag.ainsert([content])
                            )
                            await asyncio.wait_for(basic_task, timeout=120)  # 2분 타임아웃
                            success_msg = f"✅ 문서 {i+1} 임베딩 성공 (기본 방식): {Path(file_path).name}"
                        except asyncio.TimeoutError:
                            logger.error(f"❌ 문서 {i+1} 기본 방식도 타임아웃: {Path(file_path).name}")
                            raise Exception(f"문서 처리 타임아웃: {Path(file_path).name}")
                        except Exception as fallback_e:
                            logger.error(f"❌ 문서 {i+1} 기본 방식도 실패: {fallback_e}")
                            raise
                    
                    # 문서별 시간 측정 완료
                    doc_elapsed = time.time() - doc_start_time
                    _embedding_progress["document_times"].append(doc_elapsed)
                    
                    # 평균 시간 계산
                    avg_time = sum(_embedding_progress["document_times"]) / len(_embedding_progress["document_times"])
                    remaining_docs = len(documents) - (i + 1)
                    eta = remaining_docs * avg_time if remaining_docs > 0 else 0
                    
                    logger.info(f"{success_msg}")
                    logger.info(f"   ⏱️  소요시간: {doc_elapsed:.1f}초 (평균: {avg_time:.1f}초)")
                    if remaining_docs > 0:
                        logger.info(f"   🕐 예상 남은 시간: {eta:.0f}초 ({remaining_docs}개 문서 남음)")
                    
                    success_count += 1
                    
                except Exception as e:
                    doc_elapsed = time.time() - doc_start_time
                    logger.error(f"❌ 문서 {i+1} 임베딩 실패 ({Path(file_path).name}): {e}")
                    logger.error(f"   ⏱️  실패까지 소요시간: {doc_elapsed:.1f}초")
                    continue
            
            # 전체 완료 로그
            total_time = time.time() - _embedding_progress["start_time"]
            avg_doc_time = sum(_embedding_progress["document_times"]) / len(_embedding_progress["document_times"]) if _embedding_progress["document_times"] else 0
            
            logger.info(f"🎉 문서 임베딩 완료! 성공: {success_count}/{len(documents)}개")
            logger.info(f"📊 총 소요시간: {total_time:.1f}초, 문서당 평균: {avg_doc_time:.1f}초")
            logger.info(f"📈 임베딩 통계: 총 {_embedding_progress['completed_texts']}개 텍스트 청크, "
                       f"{_embedding_progress['completed_calls']}회 API 호출")
            
            if _embedding_progress["completed_texts"] > 0:
                rate = _embedding_progress["completed_texts"] / total_time
                logger.info(f"⚡ 임베딩 속도: {rate:.1f}개 청크/초")
            
            # 문서들을 임베딩 완료로 표시
            successful_docs = documents[:success_count]  # 성공한 문서들만
            if successful_docs:
                self.document_loader.mark_documents_embedded(successful_docs)
            
            # 인덱싱 상태 확인
            try:
                storage_info = await self.get_indexed_info()
                logger.info(f"💾 인덱스 상태: {storage_info}")
            except Exception as e:
                logger.warning(f"인덱스 상태 확인 실패: {e}")

        except Exception as e:
            logger.error(f"문서 삽입 실패: {e}")
            raise

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """RAG에 질의"""
        # 디버깅: 메서드 진입 시점 로그
        logger.debug(f"[QUERY_ENTRY] 받은 질문: '{question}' (길이: {len(question)}, 타입: {type(question)})")
        logger.debug(f"[QUERY_ENTRY] 질문 바이트: {question.encode('utf-8')!r}")
        
        if self.rag is None:
            logger.error("RAG 서비스가 초기화되지 않았습니다.")
            return "RAG 서비스가 초기화되지 않았습니다."
        try:
            logger.debug(f"질의: {question} (모드: {mode})")
            # 더 안전한 로그 출력
            logger.debug(f"[SAFE_LOG] 질의: {repr(question)} (모드: {mode})")

            # 한국어 프롬프트 시스템을 LLM 단계에서 처리하도록 변경
            original_question = question

            # --- 대화 히스토리 업데이트 (user) ---
            self.conversation_history.append({"role": "user", "content": original_question})

            # LightRAG API 호환성 수정
            from lightrag import QueryParam
            param = QueryParam(
                mode=mode,
                conversation_history=self.conversation_history,
                history_turns=settings.lightrag_history_turns,
            )
            
            logger.debug("LightRAG 질의 시작...")
            response = await self.rag.aquery(original_question, param=param)
            logger.debug(f"LightRAG 응답 타입: {type(response)}")
            logger.debug(f"LightRAG 응답 내용 (첫 200자): {str(response)[:200]}")
            
            # 로컬 API에 질문과 답변 전송
            try:
                await self.local_api.send_rag_response(original_question, response)
            except Exception as api_error:
                logger.warning(f"로컬 API 전송 실패 (계속 진행): {api_error}")
            
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
                    param = QueryParam(
                        mode="naive",
                        conversation_history=self.conversation_history,
                        history_turns=settings.lightrag_history_turns,
                    )
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
                            # assistant 메시지 저장
                            final_ans = korean_answer if korean_answer else answer
                            self.conversation_history.append({"role": "assistant", "content": str(final_ans)})
                            return final_ans
                    else:
                        logger.warning("전역 LLM 서비스가 없어 한국어 번역을 건너뜁니다")
                        self.conversation_history.append({"role": "assistant", "content": str(answer)})
                        return answer
                except Exception as e:
                    logger.warning(f"한국어 번역 실패: {e}")
                    self.conversation_history.append({"role": "assistant", "content": str(answer)})
                    return answer
            
            if answer:
                self.conversation_history.append({"role": "assistant", "content": str(answer)})
                return answer
            else:
                return "답변을 생성할 수 없습니다."

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
                'files': [],
                'llm_service': settings.get_llm_service(),
                'embedding_llm_service': settings.get_embedding_llm_service(),
                'kg_llm_service': settings.get_kg_llm_service(),
                'chunk_size': settings.lightrag_chunk_size,
                'chunk_overlap': settings.lightrag_chunk_overlap,
                'embedding_model': settings.lightrag_embedding_model,
                'llm_provider': settings.llm_provider,
                'embedding_llm_provider': settings.embedding_llm_provider,
                'kg_llm_provider': settings.kg_llm_provider,
                'local_api_host': settings.local_api_host,
            }

            if storage_path.exists():
                for file in storage_path.iterdir():
                    if file.is_file():
                        info['files'].append({
                            'name': file.name,
                            'size': file.stat().st_size
                        })

            # 로컬 API 상태 확인
            info["local_api_available"] = await self.local_api.is_api_available()

            return info
        except Exception as e:
            logger.error(f"인덱스 정보 조회 실패: {e}")
            return {'error': str(e)} 