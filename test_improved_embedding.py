#!/usr/bin/env python3
"""개선된 임베딩 시스템 테스트"""
import asyncio
import sys
import time
from src.Service.rag_service import RAGService
from src.Service.document_loader import DocumentLoader
from src.Config.config import settings
from loguru import logger

async def test_timeout_configuration():
    """타임아웃 설정 테스트"""
    logger.info("=" * 50)
    logger.info("🔧 타임아웃 설정 확인")
    logger.info("=" * 50)
    
    logger.info(f"📍 일반 LLM 타임아웃: {settings.llm_timeout}초")
    logger.info(f"📍 문서 임베딩 타임아웃: {settings.embedding_timeout}초")
    logger.info(f"📍 Knowledge Graph 타임아웃: {settings.kg_timeout}초")
    
    logger.info(f"\n💡 타임아웃 값은 .env 파일에서 수정 가능합니다:")
    logger.info("   - LLM_TIMEOUT (기본: 120)")
    logger.info("   - EMBEDDING_TIMEOUT (기본: 300)")
    logger.info("   - KG_TIMEOUT (기본: 180)")

async def test_embedding_with_progress():
    """진행률 표시가 개선된 임베딩 테스트"""
    logger.info("\n" + "=" * 50)
    logger.info("📊 임베딩 진행률 테스트")
    logger.info("=" * 50)
    
    try:
        # RAG 서비스 초기화
        logger.info("🚀 RAG 서비스 초기화 중...")
        rag_service = await RAGService.create()
        
        # 문서 로드
        logger.info("📄 문서 로드 중...")
        loader = DocumentLoader()
        documents = loader.load_documents(only_new=True)
        
        if not documents:
            logger.info("✅ 모든 문서가 이미 임베딩되어 있습니다.")
            
            # 임베딩 상태 표시
            status = loader.get_embedding_status()
            logger.info(f"📈 임베딩 상태:")
            logger.info(f"   - 총 임베딩 파일: {status['embedded_files_count']}개")
            logger.info(f"   - 총 파일 크기: {status['total_size_mb']:.2f} MB")
            return
        
        logger.info(f"📚 {len(documents)}개 신규/변경 문서 발견")
        
        # 임베딩 실행
        start_time = time.time()
        await rag_service.insert_documents(documents)
        elapsed = time.time() - start_time
        
        logger.info(f"\n✅ 임베딩 완료!")
        logger.info(f"⏱️  총 소요시간: {elapsed:.1f}초")
        logger.info(f"📊 평균 처리시간: {elapsed/len(documents):.1f}초/문서")
        
    except Exception as e:
        logger.error(f"❌ 테스트 중 오류 발생: {e}")
        logger.exception("상세 오류:")

async def test_graceful_shutdown():
    """Graceful shutdown 테스트"""
    logger.info("\n" + "=" * 50)
    logger.info("🛑 Graceful Shutdown 테스트")
    logger.info("=" * 50)
    logger.info("💡 이 테스트는 10초 동안 실행됩니다.")
    logger.info("   Ctrl+C를 눌러 중단 테스트를 해보세요.")
    
    try:
        # 10초 대기 (중단 테스트용)
        for i in range(10):
            logger.info(f"⏳ 대기 중... {10-i}초 남음")
            await asyncio.sleep(1)
        logger.info("✅ 정상 완료")
    except asyncio.CancelledError:
        logger.warning("⚠️  작업이 취소되었습니다.")
    except KeyboardInterrupt:
        logger.warning("⚠️  키보드 인터럽트")

async def test_llm_service_health():
    """LLM 서비스 상태 확인"""
    logger.info("\n" + "=" * 50)
    logger.info("🏥 LLM 서비스 상태 확인")
    logger.info("=" * 50)
    
    services = {
        '대화용': settings.get_llm_service(),
        '임베딩용': settings.get_embedding_llm_service(),
        'KG용': settings.get_kg_llm_service()
    }
    
    for name, service in services.items():
        logger.info(f"\n📍 {name} LLM 서비스: {service}")
        
        if service == "ollama":
            logger.info(f"   - 호스트: {settings.ollama_host}")
            logger.info(f"   - 모델: {settings.ollama_model}")
        elif service == "local":
            logger.info(f"   - API 호스트: {settings.local_api_host}")
        elif service == "openrouter":
            logger.info(f"   - 모델: {settings.openrouter_model}")

async def main():
    """메인 테스트 실행"""
    logger.info("🧪 개선된 임베딩 시스템 테스트 시작")
    logger.info("=" * 70)
    
    try:
        # 1. 타임아웃 설정 확인
        await test_timeout_configuration()
        
        # 2. LLM 서비스 상태
        await test_llm_service_health()
        
        # 3. 임베딩 테스트
        await test_embedding_with_progress()
        
        # 4. Graceful shutdown 테스트 (선택적)
        if "--test-shutdown" in sys.argv:
            await test_graceful_shutdown()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  테스트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n❌ 테스트 실패: {e}")
        logger.exception("상세 오류:")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 프로그램 종료")
        sys.exit(0) 