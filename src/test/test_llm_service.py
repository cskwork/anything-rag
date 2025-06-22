"""OllamaService 테스트 (Mocked)"""
import types
import sys
import pytest
import asyncio

from importlib import reload

# Mock Ollama Module ---------------------------------------------------------

class MockOllamaClient:
    """Ollama.Client 대체 모의 객체"""

    def __init__(self, host=None):
        self.host = host

    # 모의 모델 리스트 반환
    def list(self):
        return {
            "models": [
                {"name": "gemma3:1b"},
                {"name": "bge-m3:latest"},
            ]
        }

    # 모의 생성 결과
    def generate(self, model, prompt, options=None):
        return {"response": f"mock-{model}-{prompt}"}

    # 모의 임베딩 결과
    def embeddings(self, model, prompt):
        # 고정 길이 벡터 반환
        return {"embedding": [0.0] * 384}

# 동적 Mock 삽입 -------------------------------------------------------------

mock_ollama = types.ModuleType("ollama")
mock_ollama.Client = MockOllamaClient
sys.modules["ollama"] = mock_ollama

# 이후 import 는 Mock 사용

from src.Service.llm_service import OllamaService  # noqa: E402  pylint: disable=wrong-import-position


aio = pytest.mark.asyncio


@aio
async def test_generate_and_embed():
    """채팅/임베딩 모델 분리 및 정상 작동 여부 테스트"""
    service = OllamaService()

    # 채팅 테스트
    prompt = "테스트"
    response = await service.generate(prompt)
    assert response.startswith("mock-gemma3:1b-"), "채팅 모델 응답 모델명 불일치"

    # 임베딩 테스트
    vec = await service.embed("hello")
    assert isinstance(vec, list) and len(vec) == 384 