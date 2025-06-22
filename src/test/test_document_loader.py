"""DocumentLoader 테스트"""
import sys
import types

# --- 무거운 라이브러리 더미 모듈 삽입 --------------------------

for _mod_name in ["pypdf", "docx", "openpyxl", "markdown"]:
    if _mod_name not in sys.modules:
        # 기본 더미 모듈 생성
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

if "markdown" in sys.modules:
    # markdown.markdown 함수 더미 구현
    def _dummy_md(text):
        return text
    sys.modules["markdown"].markdown = _dummy_md

# pypdf.PdfReader 더미
if "pypdf" in sys.modules:
    class _DummyPdfReader:
        def __init__(self, *_, **__):
            self.pages = []

    sys.modules["pypdf"].PdfReader = _DummyPdfReader

# docx.Document 더미
if "docx" in sys.modules:
    def _dummy_docx_document(*_, **__):
        return types.SimpleNamespace(paragraphs=[], tables=[])

    sys.modules["docx"].Document = _dummy_docx_document

# openpyxl.load_workbook 더미
if "openpyxl" in sys.modules:
    def _dummy_workbook(*_, **__):
        # 간단한 더미 workbook 객체 반환
        Sheet = types.SimpleNamespace(iter_rows=lambda values_only=True: [])
        wb = types.SimpleNamespace(sheetnames=[], __enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None)
        wb.__dict__["__iter__"] = lambda self: iter([])
        wb.Sheet = Sheet
        return wb

    sys.modules["openpyxl"].load_workbook = _dummy_workbook

# -----------------------------------------------------------

import tempfile
from pathlib import Path

from src.Service.document_loader import DocumentLoader

def test_load_text_file(tmp_path: Path):
    """텍스트 파일 로드 테스트"""
    sample_text = "안녕하세요, 테스트입니다."
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text, encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0]["content"] == sample_text
    assert docs[0]["type"] == ".txt"

def test_document_stats(tmp_path: Path):
    """문서 통계 함수 테스트"""
    # 샘플 파일 2개 작성 (txt, md)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("hello", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_documents(tmp_path)
    stats = loader.get_document_stats(docs)

    assert stats["total_documents"] == 2
    assert stats["by_type"].get(".txt", 0) == 1
    assert stats["by_type"].get(".md", 0) == 1 