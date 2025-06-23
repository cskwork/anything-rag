"""파일 임베딩 상태 추적 유틸리티"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
from loguru import logger
from src.Config.config import settings


class FileTracker:
    """파일의 임베딩 상태를 추적하는 클래스"""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or settings.lightrag_working_dir
        self.metadata_file = self.storage_dir / "embedded_files.json"
        self.metadata: Dict[str, Dict] = {}
        self.load_metadata()
    
    def load_metadata(self):
        """메타데이터 파일에서 임베딩 상태 로드"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.debug(f"임베딩된 파일 메타데이터 로드: {len(self.metadata)}개 파일")
            else:
                self.metadata = {}
                logger.debug("새로운 임베딩 메타데이터 파일 생성")
        except Exception as e:
            logger.error(f"메타데이터 로드 실패: {e}")
            self.metadata = {}
    
    def save_metadata(self):
        """메타데이터를 파일에 저장"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.debug(f"임베딩 메타데이터 저장: {len(self.metadata)}개 파일")
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일의 MD5 해시값 계산"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"파일 해시 계산 실패 {file_path}: {e}")
            return ""
    
    def is_file_changed(self, file_path: Path) -> bool:
        """파일이 변경되었는지 확인"""
        file_key = str(file_path.relative_to(settings.input_dir))
        
        # 메타데이터에 없으면 신규 파일
        if file_key not in self.metadata:
            return True
        
        # 파일이 존재하지 않으면 변경된 것으로 간주
        if not file_path.exists():
            return True
        
        # 현재 파일 정보
        current_hash = self.get_file_hash(file_path)
        current_mtime = file_path.stat().st_mtime
        
        # 저장된 정보와 비교
        stored_info = self.metadata[file_key]
        stored_hash = stored_info.get('hash', '')
        stored_mtime = stored_info.get('mtime', 0)
        
        # 해시나 수정 시간이 다르면 변경된 것
        changed = (current_hash != stored_hash) or (current_mtime != stored_mtime)
        
        if changed:
            logger.debug(f"파일 변경 감지: {file_key}")
        
        return changed
    
    def mark_file_embedded(self, file_path: Path):
        """파일을 임베딩 완료로 표시"""
        try:
            file_key = str(file_path.relative_to(settings.input_dir))
            
            self.metadata[file_key] = {
                'hash': self.get_file_hash(file_path),
                'mtime': file_path.stat().st_mtime,
                'embedded_at': datetime.now().isoformat(),
                'size': file_path.stat().st_size
            }
            
            self.save_metadata()
            logger.debug(f"파일 임베딩 완료 표시: {file_key}")
        except Exception as e:
            logger.error(f"파일 임베딩 상태 저장 실패 {file_path}: {e}")
    
    def get_new_or_changed_files(self, all_files: list) -> list:
        """신규 또는 변경된 파일만 필터링"""
        new_files = []
        for file_info in all_files:
            file_path = Path(file_info['path'])
            if self.is_file_changed(file_path):
                new_files.append(file_info)
        
        logger.info(f"전체 {len(all_files)}개 파일 중 {len(new_files)}개 파일이 신규/변경됨")
        return new_files
    
    def get_embedded_files_count(self) -> int:
        """임베딩된 파일 수 반환"""
        return len(self.metadata)
    
    def clear_metadata(self):
        """모든 메타데이터 삭제 (초기화)"""
        self.metadata = {}
        try:
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("임베딩 메타데이터 초기화 완료")
        except Exception as e:
            logger.error(f"메타데이터 초기화 실패: {e}")
    
    def remove_missing_files(self):
        """존재하지 않는 파일들을 메타데이터에서 제거"""
        to_remove = []
        for file_key in self.metadata.keys():
            file_path = settings.input_dir / file_key
            if not file_path.exists():
                to_remove.append(file_key)
        
        for file_key in to_remove:
            del self.metadata[file_key]
            logger.debug(f"삭제된 파일 메타데이터 제거: {file_key}")
        
        if to_remove:
            self.save_metadata()
            logger.info(f"{len(to_remove)}개 삭제된 파일의 메타데이터 제거")
    
    def get_status_info(self) -> Dict:
        """현재 상태 정보 반환"""
        return {
            'embedded_files_count': len(self.metadata),
            'metadata_file': str(self.metadata_file),
            'metadata_exists': self.metadata_file.exists(),
            'last_files': list(self.metadata.keys())[-5:] if self.metadata else []
        } 