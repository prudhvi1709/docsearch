#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.31.0",
#     "openai>=1.0.0",
#     "python-dotenv>=1.0.0",
#     "pypdf>=4.2.0",
#     "tiktoken>=0.5.0",
# ]
# ///
"""
Embed documents from data/ into Cloudflare Vectorize using OpenAI embeddings via Worker
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import hashlib
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Advanced document embedder using Cloudflare Worker with sophisticated chunking"""
    
    def __init__(self, openai_api_key: str, worker_url: str = None, max_tokens: int = 7000, progress_file: str = "embedding_progress.json"):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.worker_url = worker_url or "https://docsearch-embedding.prudhvi-krovvidi.workers.dev"
        self.session = requests.Session()
        self.max_tokens = max_tokens  # Conservative buffer below the 8192 limit
        self.progress_file = Path(progress_file)
        
        # Initialize tiktoken encoder for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            # Fallback to cl100k_base encoding if model-specific encoding fails
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Load progress tracking
        self.processed_files = self.load_progress()
    
    def load_progress(self) -> Set[str]:
        """Load progress from file"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    processed = set(data.get('processed_files', []))
                    logger.info(f"Loaded progress: {len(processed)} files already processed")
                    return processed
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
        return set()
    
    def save_progress(self):
        """Save progress to file"""
        try:
            data = {
                'processed_files': list(self.processed_files),
                'last_updated': str(Path().cwd())  # Just a timestamp placeholder
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def mark_file_processed(self, file_path: Path):
        """Mark a file as successfully processed"""
        file_key = str(file_path.relative_to(file_path.parent.parent))  # Relative path from data dir parent
        self.processed_files.add(file_key)
        self.save_progress()
    
    def is_file_processed(self, file_path: Path) -> bool:
        """Check if a file has already been processed"""
        file_key = str(file_path.relative_to(file_path.parent.parent))  # Relative path from data dir parent
        return file_key in self.processed_files
    
    def read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            logger.warning(f"Failed to read PDF {file_path}: {e}")
            return ""
    
    def read_text_file(self, file_path: Path) -> str:
        """Read text from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read text file {file_path}: {e}")
            return ""
    
    def read_file_content(self, file_path: Path) -> str:
        """Read content from file based on extension"""
        if file_path.suffix.lower() == '.pdf':
            return self.read_pdf(file_path)
        else:
            return self.read_text_file(file_path)
    
    def count_tokens(self, text: str) -> int:
        """Accurately count tokens using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback to rough estimation
            return len(text) // 4
    
    def chunk_text_recursive(self, text: str, max_tokens: int = None) -> List[str]:
        """Recursively split text into chunks that fit within token limits"""
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Check if text fits within limit
        token_count = self.count_tokens(text)
        if token_count <= max_tokens:
            return [text] if text.strip() else []
        
        # Try different splitting strategies in order of preference
        chunks = []
        
        # Strategy 1: Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 1:
            current_chunk = ""
            for paragraph in paragraphs:
                # Try adding this paragraph
                test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it exists
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Recursively chunk the paragraph if it's too large
                    paragraph_chunks = self.chunk_text_recursive(paragraph, max_tokens)
                    chunks.extend(paragraph_chunks)
                    current_chunk = ""
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Strategy 2: Split by single newlines
        lines = text.split('\n')
        if len(lines) > 1:
            current_chunk = ""
            for line in lines:
                test_chunk = current_chunk + ("\n" if current_chunk else "") + line
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Recursively chunk the line if it's too large
                    line_chunks = self.chunk_text_recursive(line, max_tokens)
                    chunks.extend(line_chunks)
                    current_chunk = ""
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Strategy 3: Split by sentences
        sentences = re.split(r'[.!?]+\s+', text)
        if len(sentences) > 1:
            current_chunk = ""
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                test_chunk = current_chunk + (" " if current_chunk else "") + sentence + "."
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Recursively chunk the sentence if it's too large
                    sentence_chunks = self.chunk_text_recursive(sentence, max_tokens)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # Strategy 4: Split by words (last resort)
        words = text.split()
        if len(words) > 1:
            current_chunk = ""
            for word in words:
                test_chunk = current_chunk + (" " if current_chunk else "") + word
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        # Even a single word is too long, truncate it
                        logger.warning(f"Single word too long, truncating: {word[:50]}...")
                        # Take as many characters as possible
                        for i in range(len(word), 0, -1):
                            if self.count_tokens(word[:i]) <= max_tokens:
                                chunks.append(word[:i])
                                break
                        current_chunk = ""
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # If we get here, even splitting by words didn't work
        # This should be very rare, but handle it gracefully
        logger.warning(f"Text chunk still too large after all splitting strategies, truncating")
        return [text[:max_tokens * 3]]  # Rough character limit as last resort
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def prepare_vectors(self, file_path: Path) -> List[Dict[str, Any]]:
        """Prepare vectors from a file (may return multiple vectors if chunking is needed)"""
        try:
            # Read file content
            content = self.read_file_content(file_path)
            if not content.strip():
                logger.warning(f"File {file_path.name} is empty, skipping")
                return []
            
            # Check if chunking is needed
            total_tokens = self.count_tokens(content)
            
            if total_tokens <= self.max_tokens:
                # Single chunk
                chunks = [content]
                logger.debug(f"File {file_path.name} fits in single chunk ({total_tokens} tokens)")
            else:
                # Multiple chunks needed
                chunks = self.chunk_text_recursive(content)
                logger.info(f"File {file_path.name} split into {len(chunks)} chunks (total {total_tokens} tokens)")
            
            vectors = []
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Validate chunk size before embedding
                chunk_tokens = self.count_tokens(chunk)
                if chunk_tokens > self.max_tokens:
                    logger.error(f"Chunk {i} still too large ({chunk_tokens} tokens), skipping this chunk")
                    continue
                    
                # Generate embedding for this chunk
                embedding = self.generate_embedding(chunk)
                
                # Create unique ID for this chunk
                if len(chunks) == 1:
                    chunk_id = content_hash[:32]
                else:
                    chunk_id = f"{content_hash[:28]}_{i:03d}"
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "file_extension": file_path.suffix,
                        "file_size": str(file_path.stat().st_size),
                        "content_hash": content_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "is_chunked": len(chunks) > 1,
                        "content_preview": chunk[:200].replace('\n', ' ').strip()
                    }
                }
                vectors.append(vector)
            
            return vectors
            
        except Exception as e:
            logger.error(f"Failed to prepare vectors for {file_path}: {e}")
            return []
    
    def send_vectors_to_worker(self, vectors: List[Dict[str, Any]]) -> bool:
        """Send vectors to Cloudflare Worker for insertion"""
        try:
            logger.info(f"Sending {len(vectors)} vectors to worker: {self.worker_url}")
            
            payload = {"vectors": vectors}
            response = self.session.post(f"{self.worker_url}/embed", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"Successfully inserted {result.get('inserted', len(vectors))} vectors")
                    return True
                else:
                    logger.error(f"Worker insertion failed: {result}")
                    return False
            else:
                logger.error(f"Worker request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending vectors to worker: {e}")
            return False
    
    def embed_documents(self, data_dir: Path, batch_size: int = 10) -> bool:
        """Embed all documents in the data directory"""
        try:
            # Find all files
            if not data_dir.exists():
                logger.error(f"Data directory {data_dir} does not exist")
                return False
            
            files = list(data_dir.glob("**/*"))
            files = [f for f in files if f.is_file()]
            
            if not files:
                logger.warning("No files found to embed")
                return True
            
            logger.info(f"Found {len(files)} files to embed: {[f.name for f in files]}")
            
            # Process files in batches
            vectors = []
            successful_count = 0
            failed_count = 0
            
            for i, file_path in enumerate(files):
                logger.info(f"Processing {file_path.name} ({i+1}/{len(files)})")
                
                # Skip if already processed
                if self.is_file_processed(file_path):
                    logger.info(f"Skipping {file_path.name} - already processed")
                    continue
                
                file_vectors = self.prepare_vectors(file_path)
                if file_vectors:
                    vectors.extend(file_vectors)
                    logger.info(f"Generated {len(file_vectors)} vector(s) for {file_path.name}")
                    # Mark as processed after successful vector generation
                    self.mark_file_processed(file_path)
                else:
                    failed_count += 1
                    continue
                
                # Send batch when it reaches batch_size or at the end
                if len(vectors) >= batch_size or i == len(files) - 1:
                    if self.send_vectors_to_worker(vectors):
                        successful_count += len(vectors)
                        logger.info(f"Successfully embedded batch of {len(vectors)} vectors")
                    else:
                        failed_count += len(vectors)
                        logger.error(f"Failed to embed batch of {len(vectors)} vectors")
                    
                    vectors = []  # Reset batch
            
            logger.info(f"Embedding complete: {successful_count} successful, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"Error during embedding process: {e}")
            return False

def main():
    """Main function"""
    import sys
    
    # Check for reset flag
    reset_progress = '--reset-progress' in sys.argv
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    worker_url = os.getenv('CLOUDFLARE_WORKER_URL')  # Optional: custom worker URL
    data_dir = Path(os.getenv('DOCSEARCH_DATA_DIR', 'data'))
    
    # Prompt for missing credentials
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API Key: ")
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Using worker URL: {worker_url or 'https://docsearch-embedding.prudhvi-krovvidi.workers.dev'}")
    logger.info(f"OpenAI API key configured: {'Yes' if openai_api_key else 'No'}")
    
    # Create embedder and run
    embedder = DocumentEmbedder(openai_api_key, worker_url)
    
    # Reset progress if requested
    if reset_progress:
        logger.info("Resetting progress - will re-process all files")
        embedder.processed_files.clear()
        embedder.save_progress()
    
    try:
        success = embedder.embed_documents(data_dir)
        if success:
            logger.info("✅ Document embedding completed successfully!")
        else:
            logger.error("❌ Document embedding completed with errors.")
            return 1
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
