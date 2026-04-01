#!/usr/bin/env python3
"""
search_app.py — Semantic Desktop Search UI
==========================================
Tkinter GUI that takes a natural language query, embeds it using the same
model as the indexer, and queries the local ChromaDB for visually
contextualized results.
"""

import os
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RESULTS = 10

# ── Main Application ─────────────────────────────────────────────────────────

class SemanticSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Semantic Desktop Search")
        self.geometry("800x600")
        self.minsize(600, 400)
        
        self.db_client = None
        self.collection = None
        self.embedder = None
        
        self._build_ui()
        self._set_status("Initializing AI model and Database...")
        
        # Load heavy models in background
        threading.Thread(target=self._init_backend, daemon=True).start()

    def _build_ui(self):
        # Top Frame: Search Bar
        top = tk.Frame(self, padx=10, pady=10)
        top.pack(fill=tk.X)
        
        tk.Label(top, text="Search context:").pack(side=tk.LEFT)
        self.query_var = tk.StringVar()
        self.search_entry = tk.Entry(top, textvariable=self.query_var, font=("Segoe UI", 12))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.search_entry.bind("<Return>", lambda e: self._on_search())
        
        self.search_btn = tk.Button(top, text="Search", bg="#1976D2", fg="white", 
                                    font=("Segoe UI", 10, "bold"), command=self._on_search, state=tk.DISABLED)
        self.search_btn.pack(side=tk.LEFT)
        
        # Middle Frame: Results
        mid = tk.Frame(self, padx=10, pady=5)
        mid.pack(fill=tk.BOTH, expand=True)
        
        # Using a Text widget since Treeview doesn't support multi-line text well
        self.results_text = tk.Text(mid, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10))
        sb = tk.Scrollbar(mid, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=sb.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tag configs for clickable links and formatting
        self.results_text.tag_configure("title", font=("Segoe UI", 12, "bold"), foreground="#1976D2")
        self.results_text.tag_configure("summary", font=("Segoe UI", 10))
        self.results_text.tag_configure("meta", font=("Segoe UI", 9, "italic"), foreground="#666666")
        self.results_text.tag_configure("link", font=("Consolas", 9, "underline"), foreground="#0000EE")
        self.results_text.tag_bind("link", "<Button-1>", self._on_link_click)
        self.results_text.tag_bind("link", "<Enter>", lambda e: self.results_text.config(cursor="hand2"))
        self.results_text.tag_bind("link", "<Leave>", lambda e: self.results_text.config(cursor=""))

        # Bottom Frame: Status
        bot = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Starting up...")
        tk.Label(bot, textvariable=self.status_var, anchor="w", font=("Segoe UI", 9)).pack(fill=tk.X, padx=5, pady=2)

    def _init_backend(self):
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
            self.db_client = chromadb.PersistentClient(path=DB_DIR)
            
            # Allow searching even if collection doesn't exist yet
            try:
                self.collection = self.db_client.get_collection(name="desktop_index")
                count = self.collection.count()
                self._set_status(f"Ready. {count} files indexed.")
            except Exception:
                self._set_status("Ready. Database is currently empty (run indexer.py first).")
                self.collection = None
                
            self.after(0, lambda: self.search_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.search_entry.focus())
        except Exception as e:
            self._set_status(f"Initialization error: {str(e)}")

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

    def _on_search(self):
        query = self.query_var.get().strip()
        if not query or not self.embedder or not self.db_client:
            return
            
        self.search_btn.config(state=tk.DISABLED)
        self._set_status(f"Searching for: {query}")
        
        threading.Thread(target=self._perform_search, args=(query,), daemon=True).start()

    def _perform_search(self, query: str):
        try:
            # Re-fetch collection in case it was created since startup
            if not self.collection:
                try:
                    self.collection = self.db_client.get_collection(name="desktop_index")
                except Exception:
                    self.after(0, self._render_empty, "Database is completely empty. Run indexer.py to crawl F:\\ first.")
                    return

            if self.collection.count() == 0:
                self.after(0, self._render_empty, "Database is empty. Run indexer.py to crawl F:\\ first.")
                return

            # Embed query
            vector = self.embedder.encode(query).tolist()
            
            # Query Chroma
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=MAX_RESULTS
            )
            
            self.after(0, self._render_results, results)
        except Exception as e:
            self._set_status(f"Error searching: {str(e)}")
            self.after(0, lambda: self.search_btn.config(state=tk.NORMAL))

    def _render_empty(self, msg: str):
        self._set_status(msg)
        self.search_btn.config(state=tk.NORMAL)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"\n  {msg}\n", "meta")
        self.results_text.config(state=tk.DISABLED)

    def _render_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if not metas:
            self.results_text.insert(tk.END, "\n  No matching files found. Try re-indexing.\n", "meta")
        else:
            for i, (meta, doc, dist) in enumerate(zip(metas, docs, distances)):
                filename = meta.get("filename", "Unknown File")
                filepath = meta.get("filepath", "")
                
                # Distance threshold hint (Chroma uses L2 by default)
                # Lower is closer. E.g. < 1.0 is very good format, > 1.5 is loose.
                score = round(dist, 2)
                
                self.results_text.insert(tk.END, f"{i+1}. {filename}\n", "title")
                self.results_text.insert(tk.END, f"AI Summary: {doc}\n", "summary")
                self.results_text.insert(tk.END, f"Path: ", "meta")
                
                # Insert clickable link
                start_idx = self.results_text.index(tk.END)
                self.results_text.insert(tk.END, filepath, "link")
                end_idx = self.results_text.index(tk.END)
                
                # Tag it so clicking opens the exact file
                # The tag name needs to be unique for each link so the event handler knows what to open
                tag_name = f"link_{i}"
                self.results_text.tag_add(tag_name, start_idx, end_idx)
                self.results_text.tag_bind(tag_name, "<Button-1>", lambda e, p=filepath: self._open_file(p))
                
                self.results_text.insert(tk.END, f"  (Score: {score})\n\n", "meta")

        self.results_text.config(state=tk.DISABLED)
        self._set_status(f"Found top {len(metas)} matches.")
        self.search_btn.config(state=tk.NORMAL)

    def _open_file(self, filepath: str):
        if os.path.exists(filepath):
            os.startfile(filepath)
        else:
            messagebox.showerror("Error", f"File no longer exists at:\n{filepath}")

    def _on_link_click(self, event):
        # We handle clicks via the dynamically created 'link_N' tags
        pass

if __name__ == "__main__":
    app = SemanticSearchApp()
    app.mainloop()
