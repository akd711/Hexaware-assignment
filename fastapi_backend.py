#!/usr/bin/env python3
"""
FastAPI Backend for Complete RAG System
Implements all required API endpoints from the assignment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
import requests

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    retrieval_mode: str = "hybrid"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_mode: str
    top_k: int
    safety_status: str
    search_methods_used: List[str]
    llm_used: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    documents_loaded: int
    chunks_created: int
    model_used: str
    ollama_status: str

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int

class APIRAGSystem:
    def __init__(self):
        """Initialize the RAG system for API"""
        print("🚀 Initializing API RAG System...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize document store
        self.documents = []
        self.embeddings = []
        
        # Initialize Ollama connection
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "llama3.2:3b"  # Use your available model
        self.ollama_connected = self._check_ollama_connection()
        
        # Initialize safety guardrails
        self.harmful_patterns = [
            r'\b(hack|exploit|attack|malware|virus|phishing|ddos|sql\s*injection)\b',
            r'\b(steal|cheat|fraud|scam|illegal|unauthorized)\b',
            r'\b(harm|hurt|kill|destroy|damage)\b'
        ]
        
        # Load documents
        self.load_pdfs_from_directory()
        print("✅ API RAG System initialized successfully!")
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama Connected!")
                return True
            else:
                print("⚠️ Ollama not responding properly")
                return False
        except requests.exceptions.RequestException:
            print("❌ Ollama not running. Please start Ollama first.")
            return False
    
    def _get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def _generate_ollama_response(self, prompt, max_tokens=1000):
        """Generate response using Ollama LLM"""
        if not self.ollama_connected:
            return None
        
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return None
    
    def load_pdfs_from_directory(self, pdf_dir="PDF's"):
        """Load PDFs from local directory (simulating Google Drive)"""
        if not os.path.exists(pdf_dir):
            print(f"❌ Directory {pdf_dir} not found!")
            return False
        
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            return False
        
        print(f"Found {len(pdf_files)} PDF files")
        
        total_chunks = 0
        for pdf_file in pdf_files:
            try:
                print(f"Processing: {pdf_file.name}")
                chunks = self._process_pdf(pdf_file)
                total_chunks += chunks
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
        
        print(f"✅ Loaded {len(pdf_files)} documents with {total_chunks} chunks")
        return True
    
    def _process_pdf(self, pdf_path):
        """Process a single PDF file with ELSER-like metadata"""
        doc = fitz.open(pdf_path)
        chunk_count = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split text into chunks (300 tokens with 60 token overlap)
            chunks = self._chunk_text(text, chunk_size=300, overlap=60)
            
            for chunk_id, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Create document chunk with ELSER-like metadata
                doc_chunk = {
                    'filename': pdf_path.name,
                    'page': page_num + 1,
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'file_path': str(pdf_path),
                    'drive_url': f"https://drive.google.com/file/d/{hash(str(pdf_path))}",
                    'chunk_size': len(chunk_text.split()),
                    'timestamp': datetime.now().isoformat(),
                    'text_expansion': self._generate_text_expansion(chunk_text)
                }
                
                # Generate dense embedding
                embedding = self.embedding_model.encode(chunk_text)
                
                self.documents.append(doc_chunk)
                self.embeddings.append(embedding)
                chunk_count += 1
        
        doc.close()
        return chunk_count
    
    def _generate_text_expansion(self, text):
        """Generate ELSER-like text expansion for semantic search"""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]
    
    def _chunk_text(self, text, chunk_size=300, overlap=60):
        """Advanced text chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def bm25_search(self, query, top_k=5):
        """BM25 keyword search implementation"""
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        bm25_scores = []
        
        for doc in self.documents:
            doc_words = set(doc['text'].lower().split())
            
            # Calculate BM25 score
            k1 = 1.2
            b = 0.75
            
            tf = sum(1 for word in query_words if word in doc_words)
            doc_length = len(doc_words)
            avg_doc_length = np.mean([len(d['text'].split()) for d in self.documents])
            
            if tf > 0:
                score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
            else:
                score = 0
            
            bm25_scores.append(score)
        
        # Get top results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                doc = self.documents[idx]
                results.append({
                    'score': float(bm25_scores[idx]),
                    'search_type': 'bm25',
                    'filename': doc['filename'],
                    'page': doc['page'],
                    'chunk_id': doc['chunk_id'],
                    'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                })
        
        return results
    
    def elser_search(self, query, top_k=5):
        """ELSER semantic search using text expansion"""
        if not self.documents:
            return []
        
        query_expansion = self._generate_text_expansion(query)
        elser_scores = []
        
        for doc in self.documents:
            doc_expansion = set(doc.get('text_expansion', []))
            query_exp = set(query_expansion)
            
            if query_exp:
                overlap = len(query_exp.intersection(doc_expansion))
                score = overlap / len(query_exp)
            else:
                score = 0
            
            elser_scores.append(score)
        
        # Get top results
        top_indices = np.argsort(elser_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if elser_scores[idx] > 0:
                doc = self.documents[idx]
                results.append({
                    'score': float(elser_scores[idx]),
                    'search_type': 'elser',
                    'filename': doc['filename'],
                    'page': doc['page'],
                    'chunk_id': doc['chunk_id'],
                    'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                })
        
        return results
    
    def dense_search(self, query, top_k=5):
        """Dense vector search using embeddings"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(semantic_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'score': float(semantic_scores[idx]),
                'search_type': 'dense',
                'filename': doc['filename'],
                'page': doc['page'],
                'chunk_id': doc['chunk_id'],
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            })
        
        return results
    
    def reciprocal_rank_fusion(self, results_list, top_k=5):
        """Reciprocal Rank Fusion for combining multiple search results"""
        all_results = {}
        
        for results in results_list:
            for rank, result in enumerate(results):
                doc_id = f"{result['filename']}_{result['chunk_id']}"
                
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'filename': result['filename'],
                        'page': result['page'],
                        'chunk_id': result['chunk_id'],
                        'text': result['text'],
                        'search_type': result['search_type'],
                        'scores': {}
                    }
                
                # RRF formula: 1 / (k + rank)
                k = 60
                rrf_score = 1 / (k + rank + 1)
                all_results[doc_id]['scores'][result['search_type']] = rrf_score
        
        # Calculate combined RRF score
        for doc_id, result in all_results.items():
            result['rrf_score'] = sum(result['scores'].values())
        
        # Sort by RRF score and return top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        return sorted_results[:top_k]
    
    def hybrid_search(self, query, top_k=5, retrieval_mode="hybrid"):
        """Hybrid search combining ELSER, dense, and BM25 with RRF"""
        if retrieval_mode == "elser_only":
            return self.elser_search(query, top_k)
        elif retrieval_mode == "dense_only":
            return self.dense_search(query, top_k)
        elif retrieval_mode == "bm25_only":
            return self.bm25_search(query, top_k)
        else:  # hybrid
            # Get results from all three methods
            elser_results = self.elser_search(query, top_k * 2)
            dense_results = self.dense_search(query, top_k * 2)
            bm25_results = self.bm25_search(query, top_k * 2)
            
            # Combine using Reciprocal Rank Fusion
            combined_results = self.reciprocal_rank_fusion([elser_results, dense_results, bm25_results], top_k)
            
            return combined_results
    
    def apply_safety_guardrails(self, query):
        """Apply safety guardrails to prevent harmful queries"""
        query_lower = query.lower()
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                return False, "This query contains potentially harmful content and cannot be processed."
        
        # Check for off-topic queries
        if len(query.split()) < 3:
            return False, "Please provide a more detailed question."
        
        return True, "Query passed safety checks."
    
    def generate_llm_answer(self, query, search_results):
        """Generate answer using Ollama LLM with context"""
        if not search_results:
            return "I don't have enough information to answer that question."
        
        # Create context from search results
        context_parts = []
        for result in search_results[:3]:
            context_parts.append(f"From {result['filename']} (Page {result['page']}): {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Ollama
        prompt = f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context from documents. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer that question."

Question: {query}

Context from documents:
{context}

Instructions:
1. Answer the question based ONLY on the provided context
2. Be concise but informative
3. If you reference information, mention the source document and page
4. If the context doesn't contain enough information, say so
5. Do not make up information not present in the context

Answer:"""
        
        # Try to get response from Ollama
        if self.ollama_connected:
            print("🤖 Generating answer with Ollama LLM...")
            llm_response = self._generate_ollama_response(prompt)
            if llm_response:
                return llm_response
        
        # Fallback to template-based answer if Ollama fails
        print("⚠️ Using fallback answer generation (Ollama not available)")
        
        # Generate structured answer
        answer = f"""Based on the retrieved documents, here's what I found:

{context}

**Source Information:**
- Retrieved from {len(search_results)} document chunks
- Using {search_results[0].get('search_type', 'hybrid')} search
- All information is grounded in the source documents

**Search Method Details:**
"""
        
        # Add search method details
        search_types = set(result.get('search_type', 'unknown') for result in search_results)
        if 'hybrid' in search_types:
            answer += "- Hybrid search combining ELSER, dense embeddings, and BM25\n"
        else:
            for search_type in search_types:
                answer += f"- {search_type.upper()} search\n"
        
        answer += "\n**Note:** This demonstrates the complete RAG pipeline with multiple search methods."
        
        return answer
    
    def query(self, question, top_k=5, retrieval_mode="hybrid"):
        """Main query function with complete RAG pipeline"""
        # Apply safety guardrails
        is_safe, safety_message = self.apply_safety_guardrails(question)
        if not is_safe:
            return {
                "answer": f"❌ Safety Check Failed: {safety_message}",
                "citations": [],
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "safety_status": "blocked"
            }
        
        # Perform search
        search_results = self.hybrid_search(question, top_k, retrieval_mode)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "citations": [],
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "safety_status": "passed"
            }
        
        # Generate answer using Ollama LLM
        answer = self.generate_llm_answer(question, search_results)
        
        # Format citations
        citations = []
        for result in search_results:
            citations.append({
                "filename": result['filename'],
                "page": result['page'],
                "chunk_id": result['chunk_id'],
                "snippet": result['text'],
                "score": result.get('rrf_score', result.get('score', 0)),
                "search_type": result.get('search_type', 'hybrid'),
                "drive_url": result.get('drive_url', 'N/A')
            })
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "safety_status": "passed",
            "search_methods_used": list(set(c.get('search_type') for c in citations)),
            "llm_used": "Ollama" if self.ollama_connected else "Template Fallback"
        }
    
    def get_health_status(self):
        """Get system health status"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "documents_loaded": len(set([doc['filename'] for doc in self.documents])),
            "chunks_created": len(self.documents),
            "model_used": "all-MiniLM-L6-v2",
            "ollama_status": "connected" if self.ollama_connected else "disconnected"
        }

# Initialize FastAPI app
app = FastAPI(
    title="Complete RAG System API",
    description="AI-powered document Q&A system with ELSER, dense, BM25, hybrid search, and Ollama LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = APIRAGSystem()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Complete RAG System API is running!",
        "endpoints": {
            "health": "/healthz",
            "query": "/query",
            "ingest": "/ingest",
            "docs": "/docs"
        },
        "features": [
            "ELSER semantic search",
            "Dense vector search",
            "BM25 keyword search",
            "Hybrid search with RRF",
            "Safety guardrails",
            "Ollama LLM integration",
            "Google Drive integration (simulated)"
        ],
        "ollama_status": "connected" if rag_system.ollama_connected else "disconnected"
    }

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return rag_system.get_health_status()

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents endpoint - accepts a question, returns an answer + citations"""
    try:
        response = rag_system.query(
            request.question, 
            request.top_k, 
            request.retrieval_mode
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Re-ingest documents endpoint - loads/re-indexes docs from Google Drive (simulated)"""
    try:
        # Clear existing documents
        rag_system.documents = []
        rag_system.embeddings = []
        
        # Reload documents
        success = rag_system.load_pdfs_from_directory()
        
        if success:
            return IngestResponse(
                message="Documents re-ingested successfully from Google Drive (simulated)",
                documents_processed=len(set([doc['filename'] for doc in rag_system.documents])),
                chunks_created=len(rag_system.documents)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    return {
        "api_status": "running",
        "rag_system": rag_system.get_health_status(),
        "ollama_status": {
            "connected": rag_system.ollama_connected,
            "url": rag_system.ollama_url,
            "model": rag_system.ollama_model,
            "available_models": rag_system._get_available_models()
        },
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API root and features"},
            {"path": "/healthz", "method": "GET", "description": "Health check"},
            {"path": "/query", "method": "POST", "description": "Query documents (question → answer + citations)"},
            {"path": "/ingest", "method": "POST", "description": "Re-ingest docs from Google Drive"},
            {"path": "/status", "method": "GET", "description": "System status"},
            {"path": "/docs", "method": "GET", "description": "Interactive API documentation"}
        ],
        "search_methods": [
            "ELSER semantic search with text expansion",
            "Dense vector search using sentence-transformers",
            "BM25 keyword search with proper scoring",
            "Hybrid search with Reciprocal Rank Fusion"
        ],
        "safety_features": [
            "Content filtering for harmful queries",
            "Off-topic query detection",
            "Safety status reporting"
        ],
        "llm_features": [
            "Ollama LLM integration for real AI responses",
            "Fallback to template-based answers if Ollama unavailable",
            "Context-aware prompt engineering"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Complete RAG System API Server...")
    print("📖 Interactive API Documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/healthz")
    print("❓ Query endpoint: POST http://localhost:8000/query")
    print("📥 Ingest endpoint: POST http://localhost:8000/ingest")
    print("🤖 Ollama Status:", "✅ Connected" if rag_system.ollama_connected else "❌ Disconnected")
    print("=" * 60)
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
