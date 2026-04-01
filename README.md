# Semantic Search

A local, AI-powered desktop search tool that uses semantic embeddings and LLM-generated summaries to let you find files by meaning, not just filename.

This worked adequately on a dated AMD RX570 8GB using Gemma-3-4B.

## How It Works

1. **Indexer** (`indexer.py`) crawls a target directory, extracts text from files, and uses a local LLM to generate concise summaries of each file's content.
2. Summaries are embedded using `sentence-transformers` and stored in a local **ChromaDB** vector database.
3. **Search App** (`search_app.py`) provides a Tkinter GUI where you type natural language queries. Your query is embedded and matched against the indexed summaries to surface the top 10 most relevant files.

## Supported File Types

| Type | Extensions |
|------|-----------|
| Plain text | `.txt`, `.md`, `.csv`, `.py`, `.js`, `.html`, `.json`, `.ini` |
| PDF | `.pdf` (via PyMuPDF) |
| Word | `.docx` (via python-docx) |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp` (via LLM vision) |

## Prerequisites

- Python 3.9+
- A local LLM backend running an OpenAI-compatible API (e.g., [KoboldCpp](https://github.com/LostRuins/koboldcpp), [Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), etc.)
- A decently capable model and associated mmproj.

## Setup

### 1. Install Dependencies

```bash```
```pip install chromadb sentence-transformers requests pymupdf python-docx```

### 2. Configure the LLM Backend
Edit the configuration section at the top of indexer.py:

# Point to your local LLM's OpenAI-compatible API endpoint
KOBOLD_API = "http://localhost:5001/api/v1/chat/completions"

# Model name label (some backends ignore this, but it's required in the request)
MODEL_NAME = "gemma-3"

# Directory to crawl and index
TARGET_DIR = "F:\\"

Compatible backends (any that expose an OpenAI-compatible /v1/chat/completions endpoint):

| Backend Default | URL |	Notes |
|------|-----------|---|
KoboldCpp| http://localhost:5001/api/v1/chat/completions	|Supports vision models for image indexing
Ollama	|http://localhost:11434/v1/chat/completions	|Set MODEL_NAME to your pulled model
LM Studio	|http://localhost:1234/v1/chat/completions	|GUI-based, easy to switch models
vLLM	|http://localhost:8000/v1/chat/completions	|High-throughput serving

For image indexing you'll want a vision-capable model (e.g., llava, gemma-3, qwen2.5-vl). For text-only indexing, any LLM works.

### 3. Run the Indexer

```python indexer.py```

This will crawl your TARGET_DIR, generate summaries via the LLM, embed them, and store everything in a local chroma_db/ directory. Progress and state are saved to indexer_state.json so re-runs only process new or changed files.

### 4. Run the Search App
```python search_app.py```

A Tkinter window will open. Type natural language queries (e.g., "budget spreadsheet from last quarter" or "photo of the beach trip") and press Enter. Results show the filename, AI-generated summary, file path, and a relevance score. Click any file path to open it directly.

------
### Notes
* Privacy-first: Everything runs locally. No data leaves your machine.
* Incremental indexing: The indexer tracks file modification times and only re-processes changed files.
* Stale cleanup: Deleted or moved files are automatically purged from the index on each run.
* Windows-only: The search app uses os.startfile() to open files, which is Windows-specific. The indexer works cross-platform.

