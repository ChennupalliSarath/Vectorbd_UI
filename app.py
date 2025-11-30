from flask import Flask, render_template, request, send_file, redirect, url_for
import os, json, uuid, io
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import fitz  # PyMuPDF for PDF extraction
from docx import Document
from datetime import datetime

# ---------------- CONFIG ----------------
CHROMA_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
client = PersistentClient(path=CHROMA_DIR)
model = SentenceTransformer(EMB_MODEL)


# ---------------- HELPERS ----------------
def get_col(name):
    """Get or create a Chroma collection."""
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def extract_text_from_file(path):
    """Extract text from txt, pdf, docx."""
    ext = os.path.splitext(path)[1].lower()
    text = ""

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        pdf = fitz.open(path)
        for p in pdf:
            text += p.get_text("text") + "\n"
    elif ext == ".docx":
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])

    return text.strip()


def get_summary():
    """Dashboard summary info"""
    collections = client.list_collections()
    total_docs = 0
    recent = []

    for c in collections:
        col = get_col(c.name)
        count = 0
        try:
            count = col.count()
        except:
            count = 0
        total_docs += count
        recent.append({"name": c.name, "count": count})

    return {
        "total_collections": len(collections),
        "total_documents": total_docs,
        "collections": recent[-5:]
    }


def safe_len(x):
    """Length that is safe for None/arrays."""
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def fetch_embedding_by_id(col, doc_id):
    """Return an embedding (list) for a single id, or a string message."""
    try:
        data = col.get(ids=[doc_id], include=["embeddings"])
        embs = data.get("embeddings")
        if embs is None or safe_len(embs) == 0:
            return "[No embedding stored]"
        emb = embs[0]
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb
    except Exception as e:
        print(f"[WARN] Could not fetch embedding for {doc_id}: {e}")
        return "[Embedding unavailable]"


def keyword_fallback(col, query_text, topk=5, where=None):
    """
    If vector query returns nothing, fallback: fetch a batch of docs and
    score them by simple keyword presence + cosine sim to the query.
    This ensures the UI never shows a blank page.
    """
    # fetch a reasonable number of docs for fallback
    try:
        data = col.get(include=["documents", "metadatas"])
    except Exception as e:
        print("[ERROR] Fallback get failed:", e)
        return []

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    # try to also get ids if present (some builds add it even if not requested)
    ids = data.get("ids") or [None] * len(docs)

    if safe_len(docs) == 0:
        return []

    # compute query embedding once
    q_emb = model.encode([query_text])[0]
    result_rows = []

    # compute a light-weight score: keyword match + cosine sim
    for doc, meta, doc_id in zip(docs, metas, ids):
        kw = query_text.lower() in (doc or "").lower()
        d_emb = model.encode([doc])[0]
        # cosine similarity
        denom = (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
        cos = float(np.dot(q_emb, d_emb) / denom) if denom != 0 else 0.0
        # combined score: keyword adds a small boost
        score = cos + (0.05 if kw else 0.0)
        result_rows.append((score, doc, meta, doc_id))

    # sort best first
    result_rows.sort(key=lambda r: r[0], reverse=True)
    result_rows = result_rows[:topk]

    # build result list (with embeddings fetched per id when available)
    out = []
    for score, doc, meta, doc_id in result_rows:
        emb = fetch_embedding_by_id(col, doc_id) if doc_id else "[Embedding unavailable]"
        out.append({
            "id": doc_id or "",
            "text": doc,
            "metadata": meta,
            "embedding": emb,
            "score": round(float(score), 3)
        })
    return out


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    stats = get_summary()
    return render_template("index.html", stats=stats)


# ---------- COLLECTION MANAGEMENT ----------
@app.route("/collections", methods=["GET", "POST"])
def collections():
    if request.method == "POST":
        name = request.form["collectionName"].strip()
        if name:
            try:
                client.create_collection(name)
            except:
                pass

    cols = [c.name for c in client.list_collections()]
    return render_template("collections.html", collections=cols)


@app.route("/delete_collection/<string:col_name>")
def delete_collection(col_name):
    try:
        client.delete_collection(col_name)
    except Exception as e:
        print("Error deleting collection:", e)
    return redirect(url_for("collections"))


@app.route("/export_collection/<string:col_name>")
def export_collection(col_name):
    """Export a collection safely as JSON"""
    col = get_col(col_name)

    try:
        data = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        return f"Error reading collection: {str(e)}", 500

    documents = data.get("documents")
    metadatas = data.get("metadatas")
    embeddings = data.get("embeddings")

    if documents is None:
        documents = []
    if metadatas is None:
        metadatas = [{} for _ in range(len(documents))]
    if embeddings is None:
        embeddings = [[] for _ in range(len(documents))]

    export_list = []
    for doc, meta, emb in zip(documents, metadatas, embeddings):
        if hasattr(emb, "tolist"):
            emb = emb.tolist()

        safe_meta = {}
        if isinstance(meta, dict):
            for k, v in meta.items():
                if isinstance(v, (np.ndarray, list)):
                    safe_meta[k] = np.array(v).tolist()
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)

        export_list.append({
            "document": doc,
            "metadata": safe_meta,
            "embedding": emb
        })

    json_bytes = json.dumps(export_list, indent=2).encode("utf-8")
    buffer = io.BytesIO(json_bytes)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{col_name}_export.json",
        mimetype="application/json"
    )


# ---------- UPLOAD ----------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    cols = [c.name for c in client.list_collections()]

    if request.method == "POST":
        col_name = request.form["collection"]
        input_type = request.form["input_type"]
        col = get_col(col_name)

        metadata_raw = request.form.get("metadata", "")
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except:
            metadata = {}

        text = ""
        if input_type == "text":
            text = request.form.get("text", "").strip()
        elif input_type == "file":
            uploaded = request.files.get("file")
            if uploaded and uploaded.filename:
                save_path = os.path.join(UPLOAD_DIR, uploaded.filename)
                uploaded.save(save_path)
                text = extract_text_from_file(save_path)

        if not text:
            return "No text provided", 400

        emb = model.encode([text])[0].tolist()
        col.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[{**metadata, "timestamp": str(datetime.now())}],
            embeddings=[emb]
        )

    return render_template("upload.html", collections=cols)


# ---------- SEARCH ----------
@app.route("/search", methods=["GET", "POST"])
def search():
    cols = [c.name for c in client.list_collections()]
    results = None
    selected_col = None

    if request.method == "POST":
        selected_col = request.form["collection"]
        query = request.form["query"].strip()
        topk = int(request.form.get("topk", 5))
        mode = request.form.get("mode", "vector")
        where_raw = request.form.get("filter", "")

        col = get_col(selected_col)

        where = None
        try:
            where = json.loads(where_raw) if where_raw else None
        except:
            where = None

        # Initialize Chroma collection index if needed
        try:
            _ = col.peek()
        except Exception as e:
            print(f"[WARN] Could not peek collection {selected_col}: {e}")

        query_emb = model.encode([query])[0].tolist()

        # Call query safely (without 'embeddings' in include; some builds reject 'ids' too)
        try:
            res = col.query(
                query_embeddings=[query_emb],
                n_results=topk,
                include=["documents", "metadatas", "distances"],  # keep minimal, ids still present in res
                where=where
            )
        except Exception as e:
            print("[ERROR] Query failed:", e)
            res = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)

        results = []
        # If vector results are empty, attempt keyword fallback (or hybrid)
        if safe_len(docs) == 0:
            print("[INFO] Vector query returned 0 results; attempting fallback keyword search.")
            results = keyword_fallback(col, query, topk=topk, where=where)
        else:
            for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
                # Keyword/hybrid handling
                keyword_match = query.lower() in (doc or "").lower()
                if mode == "keyword" and not keyword_match:
                    continue
                if mode == "hybrid" and not keyword_match:
                    continue

                emb = fetch_embedding_by_id(col, doc_id) if doc_id else "[Embedding unavailable]"
                score = 1 / (1 + dist)
                results.append({
                    "id": doc_id or "",
                    "text": doc,
                    "metadata": meta,
                    "embedding": emb,
                    "score": round(score, 3)
                })

    return render_template(
        "search.html",
        collections=cols,
        results=results,
        selected_col=selected_col
    )


# ---------- DELETE DOCUMENT ----------
@app.route("/delete_document/<string:col_name>/<string:doc_id>")
def delete_document(col_name, doc_id):
    col = get_col(col_name)
    try:
        col.delete(ids=[doc_id])
    except Exception as e:
        print("Error deleting document:", e)
    return redirect(url_for("search"))


# ---------- VISUALIZATION ----------
@app.route("/visualize")
def visualize():
    cols = client.list_collections()

    all_emb = []
    all_labels = []
    colors = []
    skipped = []

    palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    for i, c in enumerate(cols):
        col = get_col(c.name)

        # Try to initialize index to avoid "Nothing found on disk"
        try:
            _ = col.peek()
        except Exception as e:
            print(f"[WARN] Skipping collection {c.name} (peek): {e}")
            skipped.append(c.name)
            continue

        try:
            data = col.get(include=["embeddings", "documents"])
        except Exception as e:
            print(f"[ERROR] Cannot read {c.name}: {e}")
            skipped.append(c.name)
            continue

        embeddings = data.get("embeddings")
        documents = data.get("documents")

        # IMPORTANT: never truth-test arrays; check length explicitly
        if embeddings is None or safe_len(embeddings) == 0:
            skipped.append(c.name)
            continue
        if documents is None or safe_len(documents) == 0:
            skipped.append(c.name)
            continue

        for e, doc in zip(embeddings, documents):
            try:
                arr = np.array(e)
                flat = arr.flatten().tolist()
                all_emb.append(flat)
                all_labels.append(f"{c.name}: {doc[:40]}")
                colors.append(palette[i % len(palette)])
            except Exception as ex:
                print(f"[WARN] Skipping bad embedding in {c.name}: {ex}")
                continue

    if safe_len(all_emb) == 0:
        msg = f"No embeddings found. Skipped: {', '.join(skipped)}"
        return render_template("visualize.html", points=[], labels=[], colors=[], message=msg)

    X = np.vstack(all_emb)
    pts = PCA(n_components=2).fit_transform(X).tolist()

    msg = f"Skipped collections (no embeddings): {', '.join(skipped)}" if skipped else ""
    return render_template("visualize.html", points=pts, labels=all_labels, colors=colors, message=msg)


# ---------- DEBUG ROUTE ----------
@app.route("/debug_vis")
def debug_vis():
    results = []
    for c in client.list_collections():
        col = get_col(c.name)
        try:
            data = col.get(include=["documents", "embeddings"])
            docs = data.get("documents")
            embs = data.get("embeddings")
            results.append({
                "collection": c.name,
                "doc_count": 0 if docs is None else len(docs),
                "emb_count": 0 if embs is None else len(embs),
                "emb_is_none": embs is None
            })
        except Exception as e:
            results.append({"collection": c.name, "error": str(e)})
    return results


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(port=5000, debug=False)
