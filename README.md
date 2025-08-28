# 🧠 Complete RAG System - AI-Powered Document Q&A

A Retrieval-Augmented Generation (RAG) system that processes local PDF documents and provides intelligent answers with citations using Ollama (local LLM).


## ✨ Features
- **🔍 Hybrid Search**: ELSER + Dense + BM25 with Reciprocal Rank Fusion
- **🤖 AI Answers**: Ollama-powered responses (llama3.2:3b or gpt-oss:20b)
- **📖 Citations**: Source documents with page numbers and snippets
- **🛡️ Safety**: Content filtering and guardrails
- **🌐 Web Interface**: Streamlit-based UI

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+
- Git
- Ollama

### Installation

1. **Clone & Setup**
```bash
git clone https://github.com/akd711/Hexaware-assignment.git
cd complete-rag-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install Ollama**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

3. **Start Ollama & Pull Model**
```bash
ollama serve
# In another terminal:
ollama pull llama3.2:3b  # Fast option
# OR
ollama pull gpt-oss:20b  # Better quality
```

4. **Add PDFs**
Place your PDF files in the `PDF's/` directory.

5. **Run System**
```bash
streamlit run complete_rag_system.py
```
Access at: `http://localhost:8501` (or next available port)

## 🔧 Configuration

1. **Copy the example environment file:**
```bash
cp env.example .env
```

2. **Edit `.env` with your settings:**
```bash
PDF_DIR=PDF's
TOP_K=5
CHUNK_SIZE=300
CHUNK_OVERLAP=60
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

3. **Add your PDFs:**
   - Create a `PDF's/` directory
   - Add your PDF documents
   - Update `PDF_DIR` in `.env` if using different directory name

## 🎯 How It Works

1. **Document Processing**: PDFs are chunked into 300-token segments with 60-token overlap
2. **Embedding Generation**: Uses sentence-transformers (all-MiniLM-L6-v2) for dense vectors
3. **Search Pipeline**: 
   - ELSER-like text expansion for semantic understanding
   - BM25 for keyword matching
   - Reciprocal Rank Fusion for hybrid results
4. **Answer Generation**: Ollama LLM processes retrieved context to generate responses
5. **Citations**: Returns source document, page, and relevant snippets

## 🐛 Troubleshooting

**Ollama Issues:**
```bash
curl http://localhost:11434/api/tags  # Check if running
ollama serve                          # Start service
ollama list                          # Check available models
```

**Port Issues:**
- Streamlit auto-assigns ports if 8501 is busy
- Check terminal output for actual URL
- Manual port: `streamlit run complete_rag_system.py --server.port 8503`

**PDF Issues:**
- Ensure PDFs are not password-protected
- Check file permissions
- Verify PDFs are valid

## 📁 Project Structure
```
├── complete_rag_system.py    # Main application
├── fastapi_backend.py        # Optional API backend
├── requirements.txt          # Dependencies
├── env.example              # Configuration template
├── sample_pdfs/             # Sample directory (replace with your PDFs)
├── README.md                # This file
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
└── CHANGELOG.md            # Version history
```

## 🎉 Success Indicators
- ✅ Ollama: "Connected"
- ✅ PDFs: "X files processed"
- ✅ Search: Working functionality
- ✅ Answers: AI-generated with citations

## 📝 Notes
- First run processes all PDFs (may take time)
- System works offline (no internet required for documents)
- Ollama models run locally on your machine
- Performance depends on your hardware and model size




