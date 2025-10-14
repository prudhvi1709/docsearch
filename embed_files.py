#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests>=2.31.0",
#     "openai>=1.0.0",
#     "python-dotenv>=1.0.0",
#     "pypdf>=4.2.0",
# ]
# ///
"""
Embed documents from dummy-data/ into Cloudflare Vectorize using OpenAI embeddings via Worker
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Simple document embedder using Cloudflare Worker"""
    
    def __init__(self, openai_api_key: str, worker_url: str = None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.worker_url = worker_url or "https://docsearch-embedding.prudhvi-krovvidi.workers.dev"
        self.session = requests.Session()
    
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
    
    def prepare_vector(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Prepare a vector from a file"""
        try:
            # Read file content
            content = self.read_file_content(file_path)
            if not content.strip():
                logger.warning(f"File {file_path.name} is empty, skipping")
                return None
            
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Create short ID (max 64 chars for Vectorize)
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            short_id = content_hash[:32]
            
            vector = {
                "id": short_id,
                "values": embedding,
                "metadata": {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "file_extension": file_path.suffix,
                    "file_size": str(file_path.stat().st_size),
                    "content_hash": content_hash,
                    "content_preview": content[:200].replace('\n', ' ').strip()
                }
            }
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to prepare vector for {file_path}: {e}")
            return None
    
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
                
                vector = self.prepare_vector(file_path)
                if vector:
                    vectors.append(vector)
                else:
                    failed_count += 1
                    continue
                
                # Send batch when it reaches batch_size or at the end
                if len(vectors) >= batch_size or i == len(files) - 1:
                    if self.send_vectors_to_worker(vectors):
                        successful_count += len(vectors)
                        logger.info(f"Successfully embedded batch of {len(vectors)} documents")
                    else:
                        failed_count += len(vectors)
                        logger.error(f"Failed to embed batch of {len(vectors)} documents")
                    
                    vectors = []  # Reset batch
            
            logger.info(f"Embedding complete: {successful_count} successful, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"Error during embedding process: {e}")
            return False

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    worker_url = os.getenv('CLOUDFLARE_WORKER_URL')  # Optional: custom worker URL
    data_dir = Path(os.getenv('DOCSEARCH_DATA_DIR', 'dummy-data'))
    
    # Prompt for missing credentials
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API Key: ")
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Using worker URL: {worker_url or 'https://docsearch-embedding.prudhvi-krovvidi.workers.dev'}")
    logger.info(f"OpenAI API key configured: {'Yes' if openai_api_key else 'No'}")
    
    # Create embedder and run
    embedder = DocumentEmbedder(openai_api_key, worker_url)
    
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
