#!/usr/bin/env python3
"""임베딩 기능 테스트 스크립트"""

import asyncio
import sys
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.Config.config import settings
from src.Service.llm_service import get_embedding_llm_service
from loguru import logger

logger.remove()
logger.add(sink=sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class EmbeddingTester:
    """임베딩 기능 테스트 클래스"""
    
    def __init__(self):
        self.embedding_service = None
        self.test_texts = [
            "안녕하세요",
            "Hello world",
            "이것은 테스트 문장입니다.",
            "This is a test sentence.",
            "머신러닝과 자연어처리는 인공지능의 핵심 기술입니다."
        ]
    
    async def initialize(self):
        """임베딩 서비스 초기화"""
        logger.info("임베딩 서비스 초기화 중...")
        try:
            self.embedding_service = await get_embedding_llm_service()
            logger.info(f"사용 중인 임베딩 서비스: {type(self.embedding_service).__name__}")
            logger.info(f"임베딩 차원: {self.embedding_service.embedding_dim}")
        except Exception as e:
            logger.error(f"임베딩 서비스 초기화 실패: {e}")
            return False
        return True
    
    async def test_single_embedding(self) -> bool:
        """단일 텍스트 임베딩 테스트"""
        logger.info("\n=== 단일 텍스트 임베딩 테스트 ===")
        
        try:
            test_text = self.test_texts[0]
            logger.info(f"테스트 텍스트: '{test_text}'")
            
            # 임베딩 생성
            embedding = await self.embedding_service.embed(test_text)
            
            # 결과 검증
            if not isinstance(embedding, list):
                logger.error(f"임베딩 결과가 리스트가 아님: {type(embedding)}")
                return False
            
            if len(embedding) == 0:
                logger.error("임베딩 벡터가 비어있음")
                return False
            
            # 벡터 품질 검사
            embedding_array = np.array(embedding)
            logger.info(f"임베딩 차원: {len(embedding)}")
            logger.info(f"벡터 범위: [{embedding_array.min():.4f}, {embedding_array.max():.4f}]")
            logger.info(f"벡터 평균: {embedding_array.mean():.4f}")
            logger.info(f"벡터 표준편차: {embedding_array.std():.4f}")
            
            # 제로 벡터 검사
            if np.allclose(embedding_array, 0):
                logger.warning("임베딩 벡터가 모두 0입니다 (더미 벡터 가능성)")
                return False
            
            logger.info("✅ 단일 텍스트 임베딩 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"단일 텍스트 임베딩 테스트 실패: {e}")
            return False
    
    async def test_batch_embedding(self) -> bool:
        """배치 텍스트 임베딩 테스트"""
        logger.info("\n=== 배치 텍스트 임베딩 테스트 ===")
        
        try:
            test_texts = self.test_texts[:3]
            logger.info(f"테스트 텍스트 개수: {len(test_texts)}")
            
            # 배치 임베딩 생성
            embeddings = await self.embedding_service.embed(test_texts)
            
            # 결과 검증
            if not isinstance(embeddings, list):
                logger.error(f"배치 임베딩 결과가 리스트가 아님: {type(embeddings)}")
                return False
            
            if len(embeddings) != len(test_texts):
                logger.error(f"임베딩 개수 불일치: 예상 {len(test_texts)}, 실제 {len(embeddings)}")
                return False
            
            # 각 임베딩 검증
            for i, embedding in enumerate(embeddings):
                if not isinstance(embedding, list) or len(embedding) == 0:
                    logger.error(f"임베딩 {i+1} 형식 오류")
                    return False
                
                embedding_array = np.array(embedding)
                if np.allclose(embedding_array, 0):
                    logger.warning(f"임베딩 {i+1}이 제로 벡터입니다")
            
            logger.info(f"✅ 배치 임베딩 테스트 통과: {len(embeddings)}개 벡터 생성")
            return True
            
        except Exception as e:
            logger.error(f"배치 임베딩 테스트 실패: {e}")
            return False
    
    async def test_similarity(self) -> bool:
        """유사도 계산 테스트"""
        logger.info("\n=== 유사도 계산 테스트 ===")
        
        try:
            # 유사한 텍스트 쌍
            similar_texts = ["안녕하세요", "안녕하십니까"]
            different_texts = ["안녕하세요", "기계학습"]
            
            # 임베딩 생성
            similar_embeddings = await self.embedding_service.embed(similar_texts)
            different_embeddings = await self.embedding_service.embed(different_texts)
            
            # 코사인 유사도 계산
            def cosine_similarity(vec1, vec2):
                vec1, vec2 = np.array(vec1), np.array(vec2)
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            similar_score = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
            different_score = cosine_similarity(different_embeddings[0], different_embeddings[1])
            
            logger.info(f"유사한 텍스트 쌍 유사도: {similar_score:.4f}")
            logger.info(f"다른 텍스트 쌍 유사도: {different_score:.4f}")
            
            # 유사한 텍스트가 더 높은 점수를 가져야 함
            if similar_score > different_score:
                logger.info("✅ 유사도 계산 테스트 통과")
                return True
            else:
                logger.warning("⚠️ 유사도 계산 결과가 예상과 다름")
                return False
                
        except Exception as e:
            logger.error(f"유사도 계산 테스트 실패: {e}")
            return False
    
    async def test_dimension_consistency(self) -> bool:
        """차원 일관성 테스트"""
        logger.info("\n=== 차원 일관성 테스트 ===")
        
        try:
            expected_dim = self.embedding_service.embedding_dim
            logger.info(f"예상 임베딩 차원: {expected_dim}")
            
            # 다양한 길이의 텍스트로 테스트
            test_cases = [
                "짧은 텍스트",
                "이것은 조금 더 긴 텍스트입니다. 여러 단어가 포함되어 있습니다.",
                "매우 긴 텍스트입니다. " * 20  # 긴 텍스트
            ]
            
            for i, text in enumerate(test_cases):
                embedding = await self.embedding_service.embed(text)
                actual_dim = len(embedding)
                
                logger.info(f"테스트 케이스 {i+1}: 텍스트 길이 {len(text)}, 임베딩 차원 {actual_dim}")
                
                if actual_dim != expected_dim:
                    logger.error(f"차원 불일치: 예상 {expected_dim}, 실제 {actual_dim}")
                    return False
            
            logger.info("✅ 차원 일관성 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"차원 일관성 테스트 실패: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        logger.info("🚀 임베딩 기능 테스트 시작")
        
        if not await self.initialize():
            return {"initialization": False}
        
        results = {}
        
        # 테스트 실행
        tests = [
            ("single_embedding", self.test_single_embedding),
            ("batch_embedding", self.test_batch_embedding),
            ("similarity", self.test_similarity),
            ("dimension_consistency", self.test_dimension_consistency)
        ]
        
        for test_name, test_func in tests:
            try:
                results[test_name] = await test_func()
            except Exception as e:
                logger.error(f"테스트 {test_name} 실행 중 오류: {e}")
                results[test_name] = False
        
        # 결과 요약
        logger.info("\n" + "="*50)
        logger.info("📊 테스트 결과 요약")
        logger.info("="*50)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ 통과" if result else "❌ 실패"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\n총 테스트: {total}개, 통과: {passed}개, 실패: {total - passed}개")
        
        if passed == total:
            logger.info("🎉 모든 임베딩 테스트 통과!")
        else:
            logger.warning("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        
        return results

async def main():
    """메인 함수"""
    try:
        tester = EmbeddingTester()
        results = await tester.run_all_tests()
        
        # 종료 코드 설정
        if all(results.values()):
            sys.exit(0)  # 모든 테스트 통과
        else:
            sys.exit(1)  # 일부 테스트 실패
            
    except KeyboardInterrupt:
        logger.info("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())