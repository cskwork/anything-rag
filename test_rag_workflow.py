#!/usr/bin/env python3
"""RAG 워크플로우 전체 테스트 스크립트"""

import asyncio
import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.Config.config import settings
from src.Service.rag_service import RAGService
from src.Service.document_loader import DocumentLoader
from loguru import logger

logger.remove()
logger.add(sink=sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class RAGWorkflowTester:
    """RAG 워크플로우 테스트 클래스"""
    
    def __init__(self):
        self.rag_service: Optional[RAGService] = None
        self.test_temp_dir = None
        self.original_input_dir = None
        
        # 테스트용 문서 내용
        self.test_documents = {
            "python_basics.txt": """
Python은 1991년 귀도 반 로섬에 의해 개발된 고급 프로그래밍 언어입니다.
Python은 간결하고 읽기 쉬운 문법을 가지고 있어 초보자도 쉽게 배울 수 있습니다.
Python은 객체지향, 함수형, 절차형 프로그래밍을 모두 지원하는 다중 패러다임 언어입니다.
Python은 웹 개발, 데이터 분석, 인공지능, 자동화 등 다양한 분야에서 사용됩니다.
            """.strip(),
            
            "machine_learning.txt": """
머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 패턴을 학습하는 기술입니다.
지도학습은 입력과 출력이 모두 주어진 데이터로 학습하는 방법입니다.
비지도학습은 출력 없이 입력 데이터만으로 패턴을 찾는 방법입니다.
강화학습은 환경과의 상호작용을 통해 보상을 최대화하는 방법을 학습합니다.
딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.
            """.strip(),
            
            "database_concepts.md": """
# 데이터베이스 개념

## 관계형 데이터베이스
관계형 데이터베이스는 테이블 형태로 데이터를 저장하는 데이터베이스입니다.
SQL(Structured Query Language)을 사용하여 데이터를 조회하고 조작합니다.

## NoSQL 데이터베이스
NoSQL 데이터베이스는 비관계형 데이터베이스로, 유연한 스키마를 가집니다.
MongoDB, Cassandra, Redis 등이 대표적인 NoSQL 데이터베이스입니다.

## 데이터베이스 설계
정규화는 데이터 중복을 최소화하고 무결성을 보장하는 설계 원칙입니다.
인덱스는 데이터 검색 속도를 향상시키는 데이터베이스 구조입니다.
            """.strip()
        }
        
        # 테스트 쿼리들
        self.test_queries = [
            "Python은 언제 개발되었나요?",
            "머신러닝의 종류는 무엇인가요?",
            "NoSQL 데이터베이스의 특징을 설명해주세요",
            "데이터베이스 정규화란 무엇인가요?",
            "딥러닝과 머신러닝의 차이점은 무엇인가요?"
        ]
    
    async def setup_test_environment(self) -> bool:
        """테스트 환경 설정"""
        logger.info("테스트 환경 설정 중...")
        
        try:
            # 임시 디렉토리 생성
            self.test_temp_dir = tempfile.mkdtemp(prefix="rag_test_")
            test_input_dir = Path(self.test_temp_dir) / "input"
            test_input_dir.mkdir()
            
            # 원본 input 디렉토리 백업
            self.original_input_dir = settings.input_dir
            
            # 테스트 문서 생성
            for filename, content in self.test_documents.items():
                file_path = test_input_dir / filename
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"테스트 문서 생성: {filename}")
            
            # 설정 임시 변경
            settings.input_dir = test_input_dir
            
            logger.info(f"테스트 디렉토리: {self.test_temp_dir}")
            return True
            
        except Exception as e:
            logger.error(f"테스트 환경 설정 실패: {e}")
            return False
    
    async def cleanup_test_environment(self):
        """테스트 환경 정리"""
        try:
            # 설정 복원
            if self.original_input_dir:
                settings.input_dir = self.original_input_dir
            
            # 임시 파일 삭제
            if self.test_temp_dir and Path(self.test_temp_dir).exists():
                import shutil
                shutil.rmtree(self.test_temp_dir)
                logger.info("테스트 환경 정리 완료")
                
        except Exception as e:
            logger.warning(f"테스트 환경 정리 중 오류: {e}")
    
    async def test_rag_service_initialization(self) -> bool:
        """RAG 서비스 초기화 테스트"""
        logger.info("\n=== RAG 서비스 초기화 테스트 ===")
        
        try:
            self.rag_service = await RAGService.create()
            
            if self.rag_service is None:
                logger.error("RAG 서비스 생성 실패")
                return False
            
            if self.rag_service.rag is None:
                logger.error("LightRAG 인스턴스 생성 실패")
                return False
            
            logger.info("✅ RAG 서비스 초기화 성공")
            return True
            
        except Exception as e:
            logger.error(f"RAG 서비스 초기화 실패: {e}")
            return False
    
    async def test_document_loading(self) -> bool:
        """문서 로딩 테스트"""
        logger.info("\n=== 문서 로딩 테스트 ===")
        
        try:
            document_loader = DocumentLoader()
            documents = await document_loader.load_documents()
            
            if not documents:
                logger.error("문서 로딩 실패 - 빈 결과")
                return False
            
            logger.info(f"로딩된 문서 수: {len(documents)}")
            
            # 문서 내용 검증
            expected_files = set(self.test_documents.keys())
            loaded_files = set()
            
            for doc_path, content in documents:
                filename = Path(doc_path).name
                loaded_files.add(filename)
                logger.info(f"문서: {filename}, 내용 길이: {len(content)} 글자")
                
                # 내용이 비어있지 않은지 확인
                if not content.strip():
                    logger.error(f"문서 {filename}의 내용이 비어있음")
                    return False
            
            # 모든 예상 파일이 로딩되었는지 확인
            missing_files = expected_files - loaded_files
            if missing_files:
                logger.error(f"누락된 파일: {missing_files}")
                return False
            
            logger.info("✅ 문서 로딩 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"문서 로딩 테스트 실패: {e}")
            return False
    
    async def test_document_insertion(self) -> bool:
        """문서 삽입 테스트"""
        logger.info("\n=== 문서 삽입 테스트 ===")
        
        try:
            # 문서 삽입
            logger.info("문서 임베딩 및 삽입 시작...")
            success = await self.rag_service.insert_documents()
            
            if not success:
                logger.error("문서 삽입 실패")
                return False
            
            # 저장소 상태 확인
            storage_dir = settings.lightrag_working_dir
            if not storage_dir.exists():
                logger.error("RAG 저장소 디렉토리가 생성되지 않음")
                return False
            
            # 저장된 파일들 확인
            storage_files = list(storage_dir.glob("*"))
            logger.info(f"생성된 저장소 파일 수: {len(storage_files)}")
            
            if len(storage_files) == 0:
                logger.error("저장소에 파일이 생성되지 않음")
                return False
            
            logger.info("✅ 문서 삽입 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"문서 삽입 테스트 실패: {e}")
            return False
    
    async def test_query_processing(self) -> bool:
        """쿼리 처리 테스트"""
        logger.info("\n=== 쿼리 처리 테스트 ===")
        
        try:
            successful_queries = 0
            
            for i, query in enumerate(self.test_queries, 1):
                logger.info(f"\n테스트 쿼리 {i}: {query}")
                
                try:
                    # 쿼리 실행
                    response = await self.rag_service.query(query, mode="naive")
                    
                    if not response or not response.strip():
                        logger.warning(f"쿼리 {i}: 빈 응답")
                        continue
                    
                    # 응답 품질 간단 체크
                    if len(response) < 10:
                        logger.warning(f"쿼리 {i}: 응답이 너무 짧음")
                        continue
                    
                    # 관련성 체크 (키워드 기반)
                    relevance_score = self._check_relevance(query, response)
                    logger.info(f"쿼리 {i}: 응답 길이 {len(response)} 글자, 관련성 점수: {relevance_score:.2f}")
                    
                    if relevance_score > 0.1:  # 최소한의 관련성
                        successful_queries += 1
                        logger.info(f"✅ 쿼리 {i} 성공")
                    else:
                        logger.warning(f"⚠️ 쿼리 {i} 관련성 낮음")
                    
                except Exception as query_error:
                    logger.error(f"쿼리 {i} 처리 오류: {query_error}")
            
            # 성공률 계산
            success_rate = successful_queries / len(self.test_queries)
            logger.info(f"\n쿼리 처리 성공률: {successful_queries}/{len(self.test_queries)} ({success_rate:.1%})")
            
            # 50% 이상 성공하면 통과
            if success_rate >= 0.5:
                logger.info("✅ 쿼리 처리 테스트 통과")
                return True
            else:
                logger.warning("⚠️ 쿼리 처리 성공률이 낮습니다")
                return False
            
        except Exception as e:
            logger.error(f"쿼리 처리 테스트 실패: {e}")
            return False
    
    def _check_relevance(self, query: str, response: str) -> float:
        """간단한 키워드 기반 관련성 점수 계산"""
        query_words = set(query.lower().replace('?', '').replace('.', '').split())
        response_words = set(response.lower().split())
        
        # 공통 단어 비율
        common_words = query_words.intersection(response_words)
        if len(query_words) == 0:
            return 0.0
        
        return len(common_words) / len(query_words)
    
    async def test_vector_storage_validation(self) -> bool:
        """벡터 저장소 검증 테스트"""
        logger.info("\n=== 벡터 저장소 검증 테스트 ===")
        
        try:
            storage_dir = settings.lightrag_working_dir
            
            # 저장소 파일들 확인
            kv_files = list(storage_dir.glob("kv_store_*.json"))
            vector_files = list(storage_dir.glob("vector_*.json"))
            
            logger.info(f"KV 저장소 파일: {len(kv_files)}개")
            logger.info(f"벡터 저장소 파일: {len(vector_files)}개")
            
            # 문서 상태 파일 확인
            doc_status_file = storage_dir / "kv_store_doc_status.json"
            if doc_status_file.exists():
                with open(doc_status_file, 'r', encoding='utf-8') as f:
                    doc_status = json.load(f)
                    logger.info(f"문서 상태 정보: {len(doc_status)}개 항목")
            
            # LLM 응답 캐시 확인
            llm_cache_file = storage_dir / "kv_store_llm_response_cache.json"
            if llm_cache_file.exists():
                with open(llm_cache_file, 'r', encoding='utf-8') as f:
                    llm_cache = json.load(f)
                    logger.info(f"LLM 응답 캐시: {len(llm_cache)}개 항목")
            
            # 최소한의 파일이 존재하는지 확인
            if len(kv_files) == 0 and len(vector_files) == 0:
                logger.error("벡터 저장소 파일이 생성되지 않음")
                return False
            
            logger.info("✅ 벡터 저장소 검증 통과")
            return True
            
        except Exception as e:
            logger.error(f"벡터 저장소 검증 실패: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        logger.info("🚀 RAG 워크플로우 전체 테스트 시작")
        
        # 테스트 환경 설정
        if not await self.setup_test_environment():
            return {"setup": False}
        
        results = {}
        
        try:
            # 테스트 실행
            tests = [
                ("rag_initialization", self.test_rag_service_initialization),
                ("document_loading", self.test_document_loading),
                ("document_insertion", self.test_document_insertion),
                ("query_processing", self.test_query_processing),
                ("vector_storage", self.test_vector_storage_validation)
            ]
            
            for test_name, test_func in tests:
                try:
                    results[test_name] = await test_func()
                    
                    # 중요한 테스트 실패 시 조기 종료
                    if test_name in ["rag_initialization", "document_insertion"] and not results[test_name]:
                        logger.error(f"중요한 테스트 {test_name} 실패로 인한 조기 종료")
                        break
                        
                except Exception as e:
                    logger.error(f"테스트 {test_name} 실행 중 오류: {e}")
                    results[test_name] = False
        
        finally:
            # 테스트 환경 정리
            await self.cleanup_test_environment()
        
        # 결과 요약
        logger.info("\n" + "="*60)
        logger.info("📊 RAG 워크플로우 테스트 결과 요약")
        logger.info("="*60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ 통과" if result else "❌ 실패"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\n총 테스트: {total}개, 통과: {passed}개, 실패: {total - passed}개")
        
        if passed == total:
            logger.info("🎉 모든 RAG 워크플로우 테스트 통과!")
        elif passed >= total * 0.7:  # 70% 이상 통과
            logger.info("✅ RAG 워크플로우가 대체로 정상 작동합니다")
        else:
            logger.warning("⚠️ RAG 워크플로우에 문제가 있을 수 있습니다")
        
        return results

async def main():
    """메인 함수"""
    try:
        tester = RAGWorkflowTester()
        results = await tester.run_all_tests()
        
        # 종료 코드 설정
        passed = sum(results.values())
        total = len(results)
        
        if passed >= total * 0.7:  # 70% 이상 통과면 성공
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())