"""로컬 API 서비스 모듈"""
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from src.Config.config import settings


class LocalApiService:
    """로컬 메시지 API 서비스"""
    
    def __init__(self):
        self.base_url = settings.local_api_host
    
    async def send_message(self, content: str, role: str = "user") -> Optional[Dict[str, Any]]:
        """메시지를 로컬 API에 전송"""
        try:
            message_data = {
                "content": content,
                "role": role,
                "time": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    json=message_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"메시지가 성공적으로 전송되었습니다: {content[:50]}...")
                        return result
                    else:
                        logger.error(f"메시지 전송 실패 (HTTP {response.status}): {await response.text()}")
                        return None
        except Exception as e:
            logger.error(f"로컬 API 메시지 전송 중 오류 발생: {e}")
            return None
    
    async def get_messages(self) -> Optional[List[Dict[str, Any]]]:
        """로컬 API에서 메시지 목록 조회"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/messages") as response:
                    if response.status == 200:
                        result = await response.json()
                        messages = result.get("messages", [])
                        logger.info(f"메시지 {len(messages)}개를 조회했습니다.")
                        return messages
                    else:
                        logger.error(f"메시지 조회 실패 (HTTP {response.status}): {await response.text()}")
                        return None
        except Exception as e:
            logger.error(f"로컬 API 메시지 조회 중 오류 발생: {e}")
            return None
    
    async def send_rag_response(self, question: str, answer: str) -> Optional[Dict[str, Any]]:
        """RAG 응답을 로컬 API에 전송"""
        # 질문과 답변을 모두 전송
        question_result = await self.send_message(question, "user")
        answer_result = await self.send_message(answer, "agent")
        
        return {
            "question": question_result,
            "answer": answer_result
        }
    
    async def is_api_available(self) -> bool:
        """로컬 API 서버 가용성 체크"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/messages", timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"로컬 API 연결 체크 실패: {e}")
            return False 