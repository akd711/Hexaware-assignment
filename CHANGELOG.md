# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- Complete RAG system with local PDF processing
- Hybrid search combining ELSER, dense embeddings, and BM25
- Ollama integration for local LLM processing
- Streamlit web interface
- FastAPI backend (optional)
- Safety guardrails and content filtering
- Citation system with source documents
- Support for multiple Ollama models (llama3.2:3b, gpt-oss:20b)
- Reciprocal Rank Fusion for search result ranking
- PDF chunking with configurable size and overlap

### Changed
- Simplified from Google Drive to local PDF processing
- Improved answer generation with natural, conversational responses
- Enhanced error handling and fallback mechanisms

### Technical Details
- Python 3.8+ compatibility
- sentence-transformers for embeddings
- PyPDF2 for PDF processing
- Streamlit for web interface
- Local in-memory search indexing

## [Unreleased]

### Planned
- Support for more document formats
- Caching and performance optimizations
- User authentication system
- Mobile-friendly interface
- Cloud storage integration options
