#!/usr/bin/env python3
"""
indexer.py — Semantic Desktop Search Backend
============================================
Crawls the specified directory (e.g., F:\), extracts text and metadata,
asks an LLM (KoboldCpp) to summarize the content, embeds the summary,
and stores it in a local ChromaDB vector database.

Requirements:
pip install chromadb sentence-transformers requests pymupdf python-docx
"""

import base64
from typing import Optional
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Optional imports for rich text extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None


# ── Configuration ────────────────────────────────────────────────────────────

TARGET_DIR = "F:\\"
DB_DIR = "chroma_db"
STATE_FILE = "indexer_state.json"

KOBOLD_API = "http://localhost:5001/api/v1/chat/completions"
MODEL_NAME = "gemma-3"  # Just a label for the request, Kobold ignores this but needs *a* string
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# File types we try to extract text from
TEXT_EXTS = {".txt", ".md", ".csv", ".py", ".js", ".html", ".json", ".ini"}
PDF_EXTS  = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Directories to always skip for speed and safety
IGNORE_DIRS = {
    "$RECYCLE.BIN", "System Volume Information", "Windows", "Program Files", "hires_texture",
    "Program Files (x86)", ".git", "node_modules", "AppData", "__pycache__"
}
# File types to completely ignore
IGNORE_EXTS = {".dll", ".exe", ".sys", ".bin", ".dat", ".cab", ".msi", ".iso", ".zip", ".rar"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Database & State ─────────────────────────────────────────────────────────

class IndexerState:
    """Tracks file mtimes so we don't re-process unchanged files."""
    def __init__(self, path: str):
        self.path = path
        self.state = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load state: {e}")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def needs_update(self, filepath: str, mtime: float) -> bool:
        return self.state.get(filepath) != mtime

    def mark_updated(self, filepath: str, mtime: float):
        self.state[filepath] = mtime


# ── AI Clients ───────────────────────────────────────────────────────────────

def get_llm_summary(filename: str, content_preview: str, is_image: bool = False, image_path: Optional[str] = None) -> str:
    """Ask KoboldCpp/Gemma3 to summarize the file.
    
    For images, sends the actual image bytes to the vision model via the
    OpenAI-compatible multimodal chat completions endpoint.
    For text files, sends a content preview as a plain text prompt.
    """
    if is_image and image_path:
        # --- Vision path: send real image data ---
        try:
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Determine MIME type from extension
            ext = Path(image_path).suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")
            
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            f"You are a file-indexing assistant. Describe the content of this image "
                            f"in 1 or 2 concise sentences for use in a search engine index. "
                            f"Focus on what is actually visible: people, objects, text, setting, or style. "
                            f"The filename is '{filename}'."
                        )
                    }
                ]
            }]
        except Exception as e:
            logging.warning(f"Could not read image file {filename} for vision: {e}")
            # Fall back to filename-only summary
            return f"Image file named '{filename}' (vision read failed)."
    elif is_image:
        # No path provided — shouldn't happen, but safe fallback
        messages = [{
            "role": "user",
            "content": f"Write a 1-sentence description for a search index about an image file named '{filename}'."
        }]
    else:
        # --- Text path: send content preview ---
        preview = content_preview[:4000]
        prompt = (
            f"You are a helpful file-indexing assistant. Summarize the following file content "
            f"in 1 or 2 concise, descriptive sentences so it can be found in a search engine.\n\n"
            f"Filename: {filename}\n"
            f"Content Preview:\n---\n{preview}\n---\n\n"
            f"Summary:"
        )
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(KOBOLD_API, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        summary = data["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        logging.error(f"LLM Summary failed for {filename}: {e}")
        return f"File named {filename} (Summary generation failed)."


# ── Text Extractors ──────────────────────────────────────────────────────────

def extract_text(path: Path) -> str:
    """Extract text from a file based on its extension."""
    ext = path.suffix.lower()
    
    try:
        if ext in TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
                
        elif ext in PDF_EXTS and fitz:
            text = []
            with fitz.open(path) as doc:
                # Just read the first 5 pages max to save time/tokens
                for page in doc[:5]:
                    text.append(page.get_text())
            return "\n".join(text)
            
        elif ext in DOCX_EXTS and docx:
            doc = docx.Document(path)
            text = [p.text for p in doc.paragraphs]
            return "\n".join(text)
            
    except Exception as e:
        logging.warning(f"Failed to extract text from {path.name}: {e}")
        
    return ""


# ── Cleanup ─────────────────────────────────────────────────────────────────

def purge_deleted_files(state: IndexerState, collection) -> int:
    """Remove entries for files that no longer exist on disk.
    
    Scans the state dict, checks each path, and deletes stale entries from
    both the IndexerState and ChromaDB. Returns the number of purged files.
    """
    stale_paths = [p for p in list(state.state.keys()) if not os.path.exists(p)]
    
    if not stale_paths:
        return 0

    logging.info(f"Purging {len(stale_paths)} deleted/moved files from index...")
    for path in stale_paths:
        doc_id = hashlib.md5(path.encode()).hexdigest()
        try:
            collection.delete(ids=[doc_id])
        except Exception as e:
            logging.warning(f"Could not delete ChromaDB entry for {path}: {e}")
        del state.state[path]
        logging.info(f"  Purged: {path}")

    state.save()
    logging.info(f"Purge complete. Removed {len(stale_paths)} stale entries.")
    return len(stale_paths)


# ── Core indexing loop ───────────────────────────────────────────────────────

def main():
    logging.info("Initializing Embedder (sentence-transformers)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    logging.info("Initializing ChromaDB...")
    db_client = chromadb.PersistentClient(path=DB_DIR)
    collection = db_client.get_or_create_collection(name="desktop_index")

    state = IndexerState(STATE_FILE)

    # Remove index entries for files that have been deleted or moved
    purge_deleted_files(state, collection)

    logging.info(f"Starting crawl of {TARGET_DIR}...")
    
    processed_count = 0
    target_path = Path(TARGET_DIR)

    for root, dirs, files in os.walk(target_path):
        # Filter directories in-place to prevent walking down them
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            path = Path(root) / file
            ext = path.suffix.lower()

            if ext in IGNORE_EXTS:
                continue

            # Only index specific types of files for now
            if not (ext in TEXT_EXTS or ext in PDF_EXTS or ext in DOCX_EXTS or ext in IMAGE_EXTS):
                continue

            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue

            abs_path = str(path.absolute())

            # Skip if already indexed and hasn't changed
            if not state.needs_update(abs_path, mtime):
                continue

            logging.info(f"Indexing: {path.name}")
            
            is_image = ext in IMAGE_EXTS
            content = ""
            if not is_image:
                content = extract_text(path)

            # 1. Summarize with LLM (pass image_path for vision inference)
            summary = get_llm_summary(
                path.name,
                content,
                is_image=is_image,
                image_path=abs_path if is_image else None
            )
            
            # 2. Embed the summary
            # (SentenceTransformer returns a numpy array, convert to list for Chroma)
            vector = embedder.encode(summary).tolist()

            # 3. Store in DB
            # Chroma requires string IDs. We'll use a hash of the path.
            doc_id = hashlib.md5(abs_path.encode()).hexdigest()
            
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[summary],
                metadatas=[{
                    "filepath": abs_path,
                    "filename": path.name,
                    "extension": ext,
                    "mtime": mtime,
                    "size": path.stat().st_size
                }]
            )

            # Update state & save periodically
            state.mark_updated(abs_path, mtime)
            processed_count += 1
            if processed_count % 10 == 0:
                state.save()
                logging.info(f"Saved state. Processed {processed_count} files...")

    # Final save
    state.save()
    logging.info(f"Crawl complete. Processed {processed_count} files.")


if __name__ == "__main__":
    main()
