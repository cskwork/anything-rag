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

# 로거 설정 (디버그 모드 활성화)
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",  # 디버그 모드 강제 활성화
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    "logs/app.log",
    level="DEBUG",  # 디버그 모드 강제 활성화
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
        self.rag_service: Optional[RAGService] = None
        self.is_initialized = False
        self.console = Console()
    
    async def initialize(self):
        """서비스 초기화"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]LightRAG 시스템 초기화 중...", total=None)

            try:
                self.rag_service = await RAGService.create()
                self.is_initialized = True
                progress.update(task, description="[green]초기화 완료!")
            except Exception as e:
                progress.update(task, description=f"[red]초기화 실패: {e}")
                self.console.print(f"[bold red]오류 발생: {e}[/bold red]")
                raise
    
    async def load_documents(self, directory: str = "input", force_reload: bool = False):
        """문서 로드 및 인덱싱
        
        Args:
            directory: 문서 디렉토리
            force_reload: True면 모든 파일 재로드, False면 신규/변경된 파일만
        """
        loader = DocumentLoader()
        
        # 임베딩 상태 정보 표시
        embedding_status = loader.get_embedding_status()
        if embedding_status['embedded_files_count'] > 0 and not force_reload:
            console.print(f"[dim]이미 임베딩된 파일: {embedding_status['embedded_files_count']}개[/dim]")
        
        documents = loader.load_documents(only_new=not force_reload)
        
        if not documents:
            if force_reload:
                console.print("[yellow]경고: input/ 폴더에 문서가 없습니다.[/yellow]")
            else:
                console.print("[green]모든 문서가 이미 임베딩되어 있습니다. 새로운 작업이 없습니다.[/green]")
                if embedding_status['embedded_files_count'] > 0:
                    console.print(f"[dim]총 {embedding_status['embedded_files_count']}개 파일이 임베딩됨[/dim]")
            return
        
        # 문서 통계 표시
        stats = loader.get_document_stats(documents)
        
        table = Table(title="로드된 문서 정보")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="magenta")
        
        if force_reload:
            table.add_row("모드", "[red]전체 재로드[/red]")
        else:
            table.add_row("모드", "[green]신규/변경 파일만[/green]")
        
        table.add_row("총 문서 수", str(stats['total_documents']))
        table.add_row("총 문자 수", f"{stats['total_characters']:,}")
        
        for doc_type, count in stats['by_type'].items():
            table.add_row(f"{doc_type} 파일", str(count))
        
        if embedding_status['embedded_files_count'] > 0:
            table.add_row("기존 임베딩", f"{embedding_status['embedded_files_count']}개 파일")
        
        console.print(table)
        
        # 문서 인덱싱
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]문서 임베딩 중...", total=None)
            
            try:
                await self.rag_service.insert_documents(documents, only_new=not force_reload)
                progress.update(task, description="[green]임베딩 완료!")
            except Exception as e:
                progress.update(task, description=f"[red]임베딩 실패: {e}")
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
                
                if question.lower() == '/reload-all':
                    await self.load_documents(force_reload=True)
                    continue
                
                if question.lower() == '/reset':
                    # 임베딩 상태 초기화 확인
                    from rich.prompt import Confirm
                    if Confirm.ask("[yellow]모든 임베딩 상태를 초기화하시겠습니까?[/yellow]"):
                        loader = DocumentLoader()
                        loader.reset_embedding_status()
                        console.print("[green]임베딩 상태 초기화 완료. 다음 로드시 모든 파일이 재처리됩니다.[/green]")
                    continue
                
                if question.lower() == '/help':
                    help_text = """
사용 가능한 명령어:
- /info       : 시스템 정보 표시
- /reload     : 신규/변경된 문서만 다시 로드
- /reload-all : 모든 문서 강제 재로드
- /reset      : 임베딩 상태 초기화
- /help       : 도움말 표시
- exit/quit/q : 종료

질의 모드:
기본적으로 hybrid 모드를 사용합니다.

파일 처리:
- 기본적으로 신규 또는 변경된 파일만 임베딩합니다
- 파일 변경은 MD5 해시와 수정 시간으로 감지합니다
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
    load_docs: bool = typer.Option(True, "--load-docs/--no-load-docs", help="시작 시 문서 로드 여부"),
    reset_embeddings: bool = typer.Option(False, "--reset-embeddings", help="시작 시 모든 임베딩 상태 초기화"),
    force_reload: bool = typer.Option(False, "--force-reload", help="모든 문서 강제 재로드")
):
    """대화형 Q&A 모드 시작"""
    async def run():
        terminal_rag = TerminalRAG()
        
        # 초기화
        await terminal_rag.initialize()
        
        # 임베딩 상태 초기화 (옵션)
        if reset_embeddings:
            from src.Service.document_loader import DocumentLoader
            loader = DocumentLoader()
            loader.reset_embedding_status()
            console.print("[yellow]임베딩 상태 초기화 완료[/yellow]")
        
        # 문서 로드
        if load_docs:
            await terminal_rag.load_documents(force_reload=force_reload)
        
        # 대화형 모드
        await terminal_rag.interactive_mode()
    
    asyncio.run(run())


@app.command()
def load(
    force_reload: bool = typer.Option(False, "--force-reload", help="모든 문서 강제 재로드"),
    reset_embeddings: bool = typer.Option(False, "--reset-embeddings", help="임베딩 상태 초기화 후 로드")
):
    """input/ 폴더의 문서를 로드하고 인덱싱"""
    async def run():
        terminal_rag = TerminalRAG()
        await terminal_rag.initialize()
        
        # 임베딩 상태 초기화 (옵션)
        if reset_embeddings:
            from src.Service.document_loader import DocumentLoader
            loader = DocumentLoader()
            loader.reset_embedding_status()
            console.print("[yellow]임베딩 상태 초기화 완료[/yellow]")
        
        await terminal_rag.load_documents(force_reload=force_reload)
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


@app.command()
def test():
    """시스템 구성 요소 테스트"""
    async def run():
        console.print("[bold cyan]LightRAG 시스템 테스트[/bold cyan]")
        
        # 1. LLM 서비스 테스트
        try:
            console.print("\n[yellow]1. LLM 서비스 테스트...[/yellow]")
            llm_service = await get_llm_service()
            test_response = await llm_service.generate("Hello, this is a test.")
            console.print(f"[green]✓ LLM 응답: {test_response[:100]}...[/green]")
        except Exception as e:
            console.print(f"[red]✗ LLM 서비스 오류: {e}[/red]")
            return
        
        # 2. 임베딩 테스트
        try:
            console.print("\n[yellow]2. 임베딩 서비스 테스트...[/yellow]")
            embedding = await llm_service.embed("test text")
            console.print(f"[green]✓ 임베딩 차원: {len(embedding)}[/green]")
        except Exception as e:
            console.print(f"[red]✗ 임베딩 서비스 오류: {e}[/red]")
            return
        
        # 3. RAG 서비스 테스트
        try:
            console.print("\n[yellow]3. RAG 서비스 초기화 테스트...[/yellow]")
            rag_service = await RAGService.create()
            console.print("[green]✓ RAG 서비스 초기화 성공[/green]")
        except Exception as e:
            console.print(f"[red]✗ RAG 서비스 오류: {e}[/red]")
            return
        
        # 4. 문서 로드 테스트
        try:
            console.print("\n[yellow]4. 문서 로드 테스트...[/yellow]")
            loader = DocumentLoader()
            documents = loader.load_documents()
            console.print(f"[green]✓ 로드된 문서 수: {len(documents)}[/green]")
            if documents:
                total_chars = sum(len(doc['content']) for doc in documents)
                console.print(f"[green]✓ 총 문자 수: {total_chars:,}[/green]")
        except Exception as e:
            console.print(f"[red]✗ 문서 로드 오류: {e}[/red]")
            return
        
        console.print("\n[bold green]모든 테스트 통과![/bold green]")
    
    asyncio.run(run())


if __name__ == "__main__":
    app() 