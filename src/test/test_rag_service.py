"""RAGService 테스트 (Mocked LightRAG)"""
import types
import sys
import pytest

import asyncio

# ---------------------------------------------------------------------------
# Dummy LightRAG 정의
# ---------------------------------------------------------------------------

class DummyLightRAG:  # pylint: disable=too-few-public-methods
    """삽입·질의 기능만 있는 간단한 LightRAG 대체"""

    def __init__(self, *_, **__):
        self.inserted = []

    async def ainsert(self, contents):
        self.inserted.extend(contents)

    async def aquery(self, question, param=None):
        return (
            f"답변({question})|docs={len(self.inserted)}|"
            f"mode={param.get('mode') if param else None}"
        )

# ---------------------------------------------------------------------------
# lightrag 더미 모듈 먼저 삽입 (rag_service import 전에 필요)
# ---------------------------------------------------------------------------

dummy_lightrag = types.ModuleType("lightrag")

# LightRAG 클래스 주입
dummy_lightrag.LightRAG = DummyLightRAG

# lightrag.llm 서브모듈 with 필요한 함수 더미 구현
llm_sub = types.ModuleType("lightrag.llm")

async def _dummy_complete(prompt, *_, **__):
    return f"complete:{prompt}"

async def _dummy_embedding(text, *_, **__):
    return [0.0] * 384

llm_sub.ollama_model_complete = _dummy_complete
llm_sub.ollama_embedding = _dummy_embedding

# OpenAI 임시 함수도 추가 (rag_service import 시 필요)

async def _dummy_openai_complete(prompt, *_, **__):
    return f"openai:{prompt}"

async def _dummy_openai_embedding(text, *_, **__):
    return [0.1] * 384

llm_sub.openai_complete_if_cache = _dummy_openai_complete
llm_sub.openai_embedding = _dummy_openai_embedding

# 삽입
dummy_lightrag.llm = llm_sub

sys.modules["lightrag"] = dummy_lightrag
sys.modules["lightrag.llm"] = llm_sub

# ---------------------------------------------------------------------------
# Patch src.Service.rag_service.LightRAG to Dummy
# ---------------------------------------------------------------------------

import importlib

rag_service_module = importlib.import_module("src.Service.rag_service")

rag_service_module.LightRAG = DummyLightRAG  # type: ignore

from src.Service.rag_service import RAGService  # noqa: E402

aio = pytest.mark.asyncio


@aio
async def test_insert_and_query():
    """문서 삽입 후 질의 결과 검증"""
    service = RAGService()

    docs = [
        {"content": "hello world", "path": "a.txt", "name": "a.txt", "type": ".txt"},
        {"content": "foo bar", "path": "b.txt", "name": "b.txt", "type": ".txt"},
    ]

    # 삽입
    await service.insert_documents(docs)

    # 질의
    response = await service.query("테스트", mode="hybrid")

    assert "docs=2" in response
    assert "테스트" in response 