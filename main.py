"""LightRAG 기반 터미널 Q&A 시스템"""
import asyncio
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger
from colorama import init as colorama_init

# Windows 터미널 색상 지원
colorama_init()

# 로거 설정
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# 모듈 임포트
from src.Config.config import settings
from src.Service.rag_service import RAGService
from src.Service.document_loader import DocumentLoader
from src.Service.llm_service import get_llm_service

# Typer 앱
app = typer.Typer(help="LightRAG 기반 문서 Q&A 시스템")
console = Console()


class TerminalRAG:
    """터미널 기반 RAG 인터페이스"""
    
    def __init__(self):
        self.rag_service = None
        self.is_initialized = False
    
    async def initialize(self):
        """서비스 초기화"""
        if not self.is_initialized:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]LightRAG 시스템 초기화 중...", total=None)
                
                try:
                    self.rag_service = RAGService()
                    self.is_initialized = True
                    progress.update(task, description="[green]초기화 완료!")
                except Exception as e:
                    progress.update(task, description=f"[red]초기화 실패: {e}")
                    raise
    
    async def load_documents(self):
        """문서 로드 및 인덱싱"""
        loader = DocumentLoader()
        documents = loader.load_documents()
        
        if not documents:
            console.print("[yellow]경고: input/ 폴더에 문서가 없습니다.[/yellow]")
            return
        
        # 문서 통계 표시
        stats = loader.get_document_stats(documents)
        
        table = Table(title="로드된 문서 정보")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="magenta")
        
        table.add_row("총 문서 수", str(stats['total_documents']))
        table.add_row("총 문자 수", f"{stats['total_characters']:,}")
        
        for doc_type, count in stats['by_type'].items():
            table.add_row(f"{doc_type} 파일", str(count))
        
        console.print(table)
        
        # 문서 인덱싱
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]문서 인덱싱 중...", total=None)
            
            try:
                await self.rag_service.insert_documents(documents)
                progress.update(task, description="[green]인덱싱 완료!")
            except Exception as e:
                progress.update(task, description=f"[red]인덱싱 실패: {e}")
                raise
    
    async def interactive_mode(self):
        """대화형 모드"""
        console.print(Panel.fit(
            "[bold cyan]LightRAG Q&A 시스템[/bold cyan]\n"
            "질문을 입력하세요. 종료하려면 'exit', 'quit', 또는 'q'를 입력하세요.",
            border_style="cyan"
        ))
        
        # Ollama 모델 확인 (memory 기반)
        if settings.get_llm_service() == "ollama":
            console.print(f"[dim]Ollama 모델 확인 중...[/dim]")
            try:
                llm_service = get_llm_service()
            except Exception as e:
                console.print(f"[yellow]Ollama 연결 실패: {e}[/yellow]")
        
        while True:
            try:
                # 질문 입력
                question = Prompt.ask("\n[bold green]질문[/bold green]")
                
                # 종료 명령 확인
                if question.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]시스템을 종료합니다.[/yellow]")
                    break
                
                # 특수 명령어
                if question.lower() == '/info':
                    info = await self.rag_service.get_indexed_info()
                    console.print(Panel(str(info), title="시스템 정보", border_style="blue"))
                    continue
                
                if question.lower() == '/reload':
                    await self.load_documents()
                    continue
                
                if question.lower() == '/help':
                    help_text = """
사용 가능한 명령어:
- /info    : 시스템 정보 표시
- /reload  : 문서 다시 로드
- /help    : 도움말 표시
- exit/quit/q : 종료

질의 모드:
기본적으로 hybrid 모드를 사용합니다.
                    """
                    console.print(Panel(help_text, title="도움말", border_style="blue"))
                    continue
                
                # RAG 질의
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("[cyan]답변 생성 중...", total=None)
                    
                    try:
                        response = await self.rag_service.query(question)
                        progress.update(task, description="[green]완료!")
                    except Exception as e:
                        progress.update(task, description=f"[red]오류: {e}")
                        continue
                
                # 답변 표시
                console.print("\n[bold blue]답변:[/bold blue]")
                console.print(Panel(Markdown(response), border_style="blue"))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Ctrl+C 감지. 종료하려면 'exit'를 입력하세요.[/yellow]")
            except Exception as e:
                console.print(f"[red]오류 발생: {e}[/red]")
                logger.exception("예외 발생")


# CLI 명령어들
@app.command()
def chat(
    load_docs: bool = typer.Option(True, "--load-docs/--no-load-docs", help="시작 시 문서 로드 여부")
):
    """대화형 Q&A 모드 시작"""
    async def run():
        terminal_rag = TerminalRAG()
        
        # 초기화
        await terminal_rag.initialize()
        
        # 문서 로드
        if load_docs:
            await terminal_rag.load_documents()
        
        # 대화형 모드
        await terminal_rag.interactive_mode()
    
    asyncio.run(run())


@app.command()
def load():
    """input/ 폴더의 문서를 로드하고 인덱싱"""
    async def run():
        terminal_rag = TerminalRAG()
        await terminal_rag.initialize()
        await terminal_rag.load_documents()
        console.print("[green]문서 로드 완료![/green]")
    
    asyncio.run(run())


@app.command()
def query(
    question: str = typer.Argument(..., help="질문"),
    mode: str = typer.Option("hybrid", help="질의 모드: naive, local, global, hybrid")
):
    """단일 질문에 대한 답변"""
    async def run():
        terminal_rag = TerminalRAG()
        await terminal_rag.initialize()
        
        with console.status("[cyan]답변 생성 중..."):
            response = await terminal_rag.rag_service.query(question, mode)
        
        console.print("\n[bold blue]답변:[/bold blue]")
        console.print(Panel(Markdown(response), border_style="blue"))
    
    asyncio.run(run())


@app.command()
def info():
    """시스템 정보 표시"""
    table = Table(title="시스템 설정")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="magenta")
    
    # LLM 서비스 정보
    llm_service = settings.get_llm_service()
    table.add_row("LLM 서비스", llm_service)
    
    if llm_service == "ollama":
        table.add_row("Ollama 호스트", settings.ollama_host)
        table.add_row("Ollama 모델", settings.ollama_model)
    elif llm_service == "openai":
        table.add_row("OpenAI 모델", settings.openai_model)
    elif llm_service == "openrouter":
        table.add_row("OpenRouter 모델", settings.openrouter_model)
    
    # 기타 설정
    table.add_row("작업 디렉토리", str(settings.lightrag_working_dir))
    table.add_row("청크 크기", str(settings.lightrag_chunk_size))
    table.add_row("청크 오버랩", str(settings.lightrag_chunk_overlap))
    table.add_row("지원 파일 형식", ", ".join(settings.supported_extensions))
    table.add_row("언어", settings.language)
    
    console.print(table)


if __name__ == "__main__":
    app() 