# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- **Windows**: `run.bat` - Automated setup and run
- **Linux/Mac**: `./run.sh` - Automated setup and run
- **Manual setup**: 
  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
  - `pip install -r requirements.txt`
  - `cp .env.example .env` (edit .env with API keys)

### Main Commands
- **Interactive chat**: `python main.py chat`
- **Load documents only**: `python main.py load`
- **Single query**: `python main.py query "your question"`
- **System info**: `python main.py info`

### Testing
- **Run tests**: `pytest src/test/`
- **Run specific test**: `pytest src/test/test_document_loader.py`
- **Test with verbose output**: `pytest -v`

## Architecture Overview

### Core Components

**Service Layer** (`src/Service/`):
- `rag_service.py`: LightRAG integration, document indexing, and query processing
- `llm_service.py`: Multi-provider LLM abstraction (OpenRouter, OpenAI, Ollama)
- `document_loader.py`: Multi-format document processing (TXT, PDF, DOCX, MD, XLSX)

**Configuration** (`src/Config/config.py`):
- Pydantic-based settings management with environment variable support
- Auto-detection of available LLM services (OpenRouter → OpenAI → Ollama)
- Configurable chunking, embedding models, and file processing limits

**Main Entry Point** (`main.py`):
- Typer-based CLI with Rich terminal interface
- Async initialization pattern for LightRAG and LLM services
- Interactive terminal with special commands (/info, /reload, /help)

### LLM Service Architecture

The system uses a factory pattern with async initialization:
- **Abstract base**: `LLMService` class with `generate()` and `embed()` methods
- **Providers**: `OllamaService`, `OpenAIService`, `OpenRouterService`
- **Auto-fallback**: Checks API keys in order: OpenRouter → OpenAI → Ollama
- **Embedding handling**: Each service manages its own embedding dimensions

### Document Processing Flow

1. **Loading**: `DocumentLoader` scans `input/` directory for supported file types
2. **Processing**: Format-specific loaders extract text content
3. **Encoding**: Auto-detection with chardet for text files
4. **Indexing**: Content passed to LightRAG for vector storage in `rag_storage/`
5. **Querying**: LightRAG handles retrieval and generation with configurable modes

### Key Patterns

- **Async Factory Pattern**: Services use `@classmethod async def create()` for initialization
- **Configuration Management**: Centralized settings with environment variable fallbacks
- **Error Handling**: Comprehensive logging with loguru, graceful degradation
- **Korean Language Support**: Auto-detection and language-specific prompting
- **Multi-modal Querying**: Support for naive, local, global, and hybrid RAG modes

### Storage Structure
- `input/`: Document source files
- `rag_storage/`: LightRAG vector database and indices
- `logs/`: Application logs with rotation
- `backup/`: Optional backup directory

### Dependencies of Note
- **LightRAG**: Core RAG functionality with graph-based retrieval
- **Typer + Rich**: Modern CLI with progress indicators and formatted output
- **Pydantic Settings**: Type-safe configuration management
- **Multiple LLM Clients**: ollama, openai, custom OpenRouter integration