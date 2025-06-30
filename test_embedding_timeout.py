#!/usr/bin/env python3
"""임베딩 타임아웃 및 graceful shutdown 테스트"""
import asyncio
import signal
import sys
from src.Service.rag_service import RAGService
from src.Service.document_loader import DocumentLoader
from loguru import logger

# 전역 변수로 진행 중인 태스크 추적
running_tasks = set()
shutdown_event = asyncio.Event()

def signal_handler(signame):
    """시그널 핸들러"""
    logger.warning(f"🛑 {signame} 시그널 수신 - 종료 프로세스 시작...")
    shutdown_event.set()

async def test_embedding_with_graceful_shutdown():
    """Graceful shutdown이 가능한 임베딩 테스트"""
    # 시그널 핸들러 등록
    loop = asyncio.get_running_loop()
    for signame in {'SIGINT', 'SIGTERM'}:
        if hasattr(signal, signame):
            loop.add_signal_handler(
                getattr(signal, signame),
                lambda: signal_handler(signame)
            )
    
    try:
        logger.info("🚀 RAG 서비스 초기화 중...")
        rag_service = await RAGService.create()
        
        logger.info("📄 문서 로드 중...")
        loader = DocumentLoader()
        documents = loader.load_documents(only_new=True)
        
        if not documents:
            logger.info("새로운 문서가 없습니다.")
            return
        
        logger.info(f"📚 {len(documents)}개 문서 발견")
        
        # 임베딩 작업을 태스크로 생성
        embed_task = asyncio.create_task(
            rag_service.insert_documents(documents)
        )
        running_tasks.add(embed_task)
        embed_task.add_done_callback(running_tasks.discard)
        
        # shutdown 이벤트 또는 태스크 완료를 기다림
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        done, pending = await asyncio.wait(
            {embed_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        
        if shutdown_task in done:
            logger.warning("⚠️  종료 시그널 감지 - 진행 중인 작업 취소...")
            
            # 진행 중인 태스크 취소
            for task in pending:
                task.cancel()
            
            # 모든 태스크 완료 대기 (짧은 시간만)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.error("⏰ 일부 태스크가 5초 내에 종료되지 않았습니다.")
            
            logger.info("✅ Graceful shutdown 완료")
        else:
            logger.info("✅ 임베딩 작업 정상 완료")
            
    except asyncio.CancelledError:
        logger.warning("작업이 취소되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        logger.exception("상세 오류:")
    finally:
        # 시그널 핸들러 제거
        for signame in {'SIGINT', 'SIGTERM'}:
            if hasattr(signal, signame):
                loop.remove_signal_handler(getattr(signal, signame))

async def main():
    """메인 함수"""
    try:
        await test_embedding_with_graceful_shutdown()
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트 감지")
    finally:
        # 남은 태스크 정리
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        if tasks:
            logger.info(f"남은 태스크 {len(tasks)}개 정리 중...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("프로그램 종료")
        sys.exit(0) 