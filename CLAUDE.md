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
- `llm_service.py`: Multi-provider LLM abstraction (Local API, OpenRouter, OpenAI, Ollama)
- `local_api_service.py`: Local message API service for RAG response logging
- `document_loader.py`: Multi-format document processing (TXT, PDF, DOCX, MD, XLSX)

**Configuration** (`src/Config/config.py`):
- Pydantic-based settings management with environment variable support
- Auto-detection of available LLM services (Local API → Ollama → OpenRouter)
- Unified provider selection via `LLM_PROVIDER` setting
- Configurable chunking, embedding models, and file processing limits

**Main Entry Point** (`main.py`):
- Typer-based CLI with Rich terminal interface
- Async initialization pattern for LightRAG and LLM services
- Interactive terminal with special commands (/info, /reload, /help)

### LLM Service Architecture

The system uses a factory pattern with async initialization:
- **Abstract base**: `LLMService` class with `generate()` and `embed()` methods
- **Providers**: `LocalLLMService`, `OllamaService`, `OpenAIService`, `OpenRouterService`
- **Auto-fallback**: Priority order: Local API → Ollama → OpenRouter → OpenAI
- **Local API**: OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`)
- **Embedding handling**: Each service manages its own embedding dimensions with fallback support

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
- **Multiple LLM Clients**: aiohttp (local), ollama, openai, custom OpenRouter integration

## LLM Provider Configuration

### Provider Selection Priority
The system automatically selects LLM providers in this order:
1. **Local API** (`LOCAL_API_HOST` configured)
2. **Ollama** (`OLLAMA_HOST` configured)
3. **OpenRouter** (`OPENROUTER_API_KEY` provided)
4. **OpenAI** (`OPENAI_API_KEY` provided)

### Configuration Options
```env
# Provider selection (auto, local, ollama, openai, openrouter)
LLM_PROVIDER=auto

# Local LLM API (OpenAI-compatible)
LOCAL_API_HOST=http://localhost:3284

# Ollama configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest

# OpenAI configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo

# OpenRouter configuration
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

### Local LLM Service Features
- **Connection Testing**: Multiple endpoint health checks
- **OpenAI Compatibility**: Standard `/v1/chat/completions` and `/v1/embeddings` endpoints
- **Embedding Fallback**: Auto-fallback to Ollama for embeddings if local API fails
- **Error Handling**: Graceful degradation to next available provider