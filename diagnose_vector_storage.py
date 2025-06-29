#!/usr/bin/env python3
"""벡터 저장소 진단 도구"""

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.Config.config import settings
from loguru import logger

logger.remove()
logger.add(sink=sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class VectorStorageDiagnostic:
    """벡터 저장소 진단 클래스"""
    
    def __init__(self):
        self.storage_dir = settings.lightrag_working_dir
        self.issues = []
        self.recommendations = []
    
    def diagnose_directory_structure(self) -> Dict[str, Any]:
        """디렉토리 구조 진단"""
        logger.info("\n=== 디렉토리 구조 진단 ===")
        
        result = {
            "exists": False,
            "readable": False,
            "writable": False,
            "files": [],
            "size_mb": 0
        }
        
        try:
            # 디렉토리 존재 확인
            if not self.storage_dir.exists():
                self.issues.append("RAG 저장소 디렉토리가 존재하지 않음")
                self.recommendations.append("python main.py load 명령어를 실행하여 문서를 로딩하세요")
                return result
            
            result["exists"] = True
            logger.info(f"✅ 저장소 디렉토리 존재: {self.storage_dir}")
            
            # 권한 확인
            result["readable"] = os.access(self.storage_dir, os.R_OK)
            result["writable"] = os.access(self.storage_dir, os.W_OK)
            
            if not result["readable"]:
                self.issues.append("저장소 디렉토리 읽기 권한 없음")
            if not result["writable"]:
                self.issues.append("저장소 디렉토리 쓰기 권한 없음")
            
            # 파일 목록 및 크기 확인
            total_size = 0
            for file_path in self.storage_dir.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    result["files"].append({
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(self.storage_dir)),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / 1024 / 1024, 2)
                    })
            
            result["size_mb"] = round(total_size / 1024 / 1024, 2)
            logger.info(f"📁 총 파일 수: {len(result['files'])}개")
            logger.info(f"💾 총 크기: {result['size_mb']} MB")
            
            return result
            
        except Exception as e:
            logger.error(f"디렉토리 구조 진단 실패: {e}")
            self.issues.append(f"directory_structure_error: {e}")
            return result
    
    def diagnose_storage_files(self, files_info: List[Dict]) -> Dict[str, Any]:
        """저장소 파일들 진단"""
        logger.info("\n=== 저장소 파일 진단 ===")
        
        result = {
            "kv_stores": [],
            "vector_stores": [],
            "graphs": [],
            "caches": [],
            "others": [],
            "empty_files": [],
            "corrupted_files": []
        }
        
        try:
            for file_info in files_info:
                file_path = self.storage_dir / file_info["relative_path"]
                file_name = file_info["name"]
                
                # 파일이 비어있는지 확인
                if file_info["size_bytes"] == 0:
                    result["empty_files"].append(file_name)
                    continue
                
                # 파일 형태별 분류
                if file_name.startswith("kv_store_"):
                    result["kv_stores"].append(self._analyze_kv_store(file_path))
                elif file_name.startswith("vector_"):
                    result["vector_stores"].append(self._analyze_vector_store(file_path))
                elif "graph" in file_name:
                    result["graphs"].append(self._analyze_graph_file(file_path))
                elif "cache" in file_name:
                    result["caches"].append(self._analyze_cache_file(file_path))
                else:
                    result["others"].append(file_info)
            
            # 결과 요약
            logger.info(f"📚 KV 저장소: {len(result['kv_stores'])}개")
            logger.info(f"🔢 벡터 저장소: {len(result['vector_stores'])}개") 
            logger.info(f"🕸️ 그래프 파일: {len(result['graphs'])}개")
            logger.info(f"💨 캐시 파일: {len(result['caches'])}개")
            logger.info(f"❓ 기타 파일: {len(result['others'])}개")
            
            if result["empty_files"]:
                logger.warning(f"⚠️ 빈 파일: {result['empty_files']}")
                self.issues.append(f"빈 파일들이 발견됨: {result['empty_files']}")
            
            if result["corrupted_files"]:
                logger.error(f"❌ 손상된 파일: {result['corrupted_files']}")
                self.issues.append(f"손상된 파일들이 발견됨: {result['corrupted_files']}")
            
            return result
            
        except Exception as e:
            logger.error(f"저장소 파일 진단 실패: {e}")
            self.issues.append(f"storage_files_error: {e}")
            return result
    
    def _analyze_kv_store(self, file_path: Path) -> Dict[str, Any]:
        """KV 저장소 파일 분석"""
        result = {
            "file": file_path.name,
            "type": "kv_store",
            "valid": False,
            "entries_count": 0,
            "sample_keys": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result["valid"] = True
            result["entries_count"] = len(data) if isinstance(data, dict) else 0
            
            if isinstance(data, dict):
                result["sample_keys"] = list(data.keys())[:5]  # 처음 5개 키만
            
            logger.info(f"  ✅ {file_path.name}: {result['entries_count']}개 항목")
            
        except json.JSONDecodeError:
            logger.error(f"  ❌ {file_path.name}: JSON 파싱 오류")
            result["valid"] = False
        except Exception as e:
            logger.error(f"  ❌ {file_path.name}: 분석 오류 - {e}")
            result["valid"] = False
        
        return result
    
    def _analyze_vector_store(self, file_path: Path) -> Dict[str, Any]:
        """벡터 저장소 파일 분석"""
        result = {
            "file": file_path.name,
            "type": "vector_store", 
            "valid": False,
            "vectors_count": 0,
            "dimensions": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result["valid"] = True
            
            if isinstance(data, dict):
                result["vectors_count"] = len(data)
                # 첫 번째 벡터의 차원 확인
                if data:
                    first_vector = next(iter(data.values()))
                    if isinstance(first_vector, list):
                        result["dimensions"] = len(first_vector)
            
            logger.info(f"  ✅ {file_path.name}: {result['vectors_count']}개 벡터, {result['dimensions']}차원")
            
        except json.JSONDecodeError:
            logger.error(f"  ❌ {file_path.name}: JSON 파싱 오류")
            result["valid"] = False
        except Exception as e:
            logger.error(f"  ❌ {file_path.name}: 분석 오류 - {e}")
            result["valid"] = False
        
        return result
    
    def _analyze_graph_file(self, file_path: Path) -> Dict[str, Any]:
        """그래프 파일 분석"""
        result = {
            "file": file_path.name,
            "type": "graph",
            "valid": False,
            "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2)
        }
        
        try:
            # 파일 형식에 따라 다른 분석
            if file_path.suffix == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                result["valid"] = True
            else:
                # 바이너리 파일인 경우 크기만 확인
                result["valid"] = file_path.stat().st_size > 0
            
            logger.info(f"  ✅ {file_path.name}: {result['size_mb']} MB")
            
        except Exception as e:
            logger.error(f"  ❌ {file_path.name}: 분석 오류 - {e}")
            result["valid"] = False
        
        return result
    
    def _analyze_cache_file(self, file_path: Path) -> Dict[str, Any]:
        """캐시 파일 분석"""
        result = {
            "file": file_path.name,
            "type": "cache",
            "valid": False,
            "entries_count": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result["valid"] = True
            result["entries_count"] = len(data) if isinstance(data, dict) else 0
            
            logger.info(f"  ✅ {file_path.name}: {result['entries_count']}개 캐시 항목")
            
        except Exception as e:
            logger.error(f"  ❌ {file_path.name}: 분석 오류 - {e}")
            result["valid"] = False
        
        return result
    
    def check_embedding_consistency(self, storage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """임베딩 일관성 확인"""
        logger.info("\n=== 임베딩 일관성 확인 ===")
        
        result = {
            "expected_dimension": settings.embedding_dim or 1024,
            "found_dimensions": [],
            "consistent": True,
            "total_vectors": 0
        }
        
        try:
            for vector_store in storage_analysis.get("vector_stores", []):
                if vector_store["valid"] and vector_store["dimensions"] > 0:
                    result["found_dimensions"].append(vector_store["dimensions"])
                    result["total_vectors"] += vector_store["vectors_count"]
            
            # 차원 일관성 확인
            unique_dimensions = set(result["found_dimensions"])
            if len(unique_dimensions) > 1:
                result["consistent"] = False
                self.issues.append(f"임베딩 차원 불일치: {unique_dimensions}")
                self.recommendations.append("rag_storage 디렉토리를 삭제하고 다시 문서를 로딩하세요")
            
            expected_dim = result["expected_dimension"]
            if unique_dimensions and expected_dim not in unique_dimensions:
                result["consistent"] = False
                self.issues.append(f"설정된 차원({expected_dim})과 실제 차원({unique_dimensions}) 불일치")
                self.recommendations.append(f".env 파일의 EMBEDDING_DIM을 {list(unique_dimensions)[0]}로 수정하세요")
            
            logger.info(f"📏 예상 차원: {expected_dim}")
            logger.info(f"🔍 발견된 차원: {unique_dimensions}")
            logger.info(f"📊 총 벡터 수: {result['total_vectors']}")
            
            if result["consistent"]:
                logger.info("✅ 임베딩 차원 일관성 OK")
            else:
                logger.warning("⚠️ 임베딩 차원 불일치 발견")
            
            return result
            
        except Exception as e:
            logger.error(f"임베딩 일관성 확인 실패: {e}")
            self.issues.append(f"embedding_consistency_error: {e}")
            return result
    
    def suggest_solutions(self):
        """해결책 제안"""
        logger.info("\n" + "="*60)
        logger.info("🔧 진단 결과 및 해결책")
        logger.info("="*60)
        
        if not self.issues:
            logger.info("✅ 벡터 저장소 상태가 정상입니다!")
            return
        
        logger.info("⚠️ 발견된 문제점:")
        for i, issue in enumerate(self.issues, 1):
            logger.info(f"  {i}. {issue}")
        
        logger.info("\n💡 추천 해결책:")
        for i, rec in enumerate(self.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # 추가 일반적인 해결책
        logger.info("\n🛠️ 일반적인 해결 방법:")
        logger.info("  • 임베딩 테스트: python test_embedding.py")
        logger.info("  • RAG 워크플로우 테스트: python test_rag_workflow.py") 
        logger.info("  • 문서 재로딩: python main.py load")
        logger.info("  • 저장소 초기화: rm -rf rag_storage && python main.py load")
        logger.info("  • Ollama 모델 확인: ollama list")
        logger.info("  • Ollama 임베딩 모델 설치: ollama pull bge-m3:latest")
    
    async def run_full_diagnosis(self) -> Dict[str, Any]:
        """전체 진단 실행"""
        logger.info("🔍 벡터 저장소 전체 진단 시작")
        
        diagnosis_result = {
            "directory": {},
            "files": {},
            "embedding": {},
            "issues_count": 0,
            "recommendations_count": 0
        }
        
        try:
            # 1. 디렉토리 구조 진단
            diagnosis_result["directory"] = self.diagnose_directory_structure()
            
            # 2. 저장소 파일들 진단 (디렉토리가 존재하는 경우에만)
            if diagnosis_result["directory"]["exists"]:
                diagnosis_result["files"] = self.diagnose_storage_files(
                    diagnosis_result["directory"]["files"]
                )
                
                # 3. 임베딩 일관성 확인
                diagnosis_result["embedding"] = self.check_embedding_consistency(
                    diagnosis_result["files"]
                )
            
            # 4. 결과 요약 및 해결책 제안
            diagnosis_result["issues_count"] = len(self.issues)
            diagnosis_result["recommendations_count"] = len(self.recommendations)
            
            self.suggest_solutions()
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"진단 실행 중 오류: {e}")
            logger.error(traceback.format_exc())
            return diagnosis_result

async def main():
    """메인 함수"""
    try:
        diagnostic = VectorStorageDiagnostic()
        result = await diagnostic.run_full_diagnosis()
        
        # 종료 코드 설정
        if result["issues_count"] == 0:
            sys.exit(0)  # 문제 없음
        else:
            sys.exit(1)  # 문제 발견
            
    except KeyboardInterrupt:
        logger.info("\n진단이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"진단 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())