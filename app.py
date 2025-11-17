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
BATCH_SIZE = 200

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
    """Dashboard stats"""
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
    col = get_col(col_name)

    try:
        data = col.get(include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        return f"Error reading collection: {str(e)}", 500

    documents = data.get("documents")
    metadatas = data.get("metadatas")
    embeddings = data.get("embeddings")

    # Handle None or NumPy arrays safely
    if documents is None or len(documents) == 0:
        documents = []
    if metadatas is None or len(metadatas) == 0:
        metadatas = [{} for _ in range(len(documents))]
    if embeddings is None or len(embeddings) == 0:
        embeddings = [[] for _ in range(len(documents))]

    export_list = []

    for doc, meta, emb in zip(documents, metadatas, embeddings):
        # Convert embedding safely
        if hasattr(emb, "tolist"):
            emb = emb.tolist()

        # Make metadata JSON safe
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

    # Handle completely empty collection
    if len(export_list) == 0:
        export_list = [{"message": "No data found in this collection"}]

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

        # Metadata
        metadata_raw = request.form.get("metadata", "")
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except:
            metadata = {}

        # Handle text vs file
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
            embeddings=[emb],
        )

    return render_template("upload.html", collections=cols)


# ---------- HYBRID SEARCH + FILTER ----------
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

        # Metadata filter
        where = None
        try:
            where = json.loads(where_raw) if where_raw else None
        except:
            where = None

        query_emb = model.encode([query])[0].tolist()

        res = col.query(
            query_embeddings=[query_emb],
            n_results=topk,
            include=["documents", "metadatas", "distances"],
            where=where
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        ids = res["ids"][0]

        # Build final result list
        results = []
        for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
            score = 1 / (1 + dist)

            # Keyword check
            keyword_match = query.lower() in doc.lower()

            if mode == "keyword" and not keyword_match:
                continue

            if mode == "hybrid" and not keyword_match:
                continue

            results.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": score
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


# ---------- VISUALIZATION (PLOTLY) ----------
@app.route("/visualize")
def visualize():
    cols = client.list_collections()

    all_emb = []
    all_labels = []
    colors = []

    palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    for i, c in enumerate(cols):
        col = get_col(c.name)

        data = col.get(include=["embeddings", "documents"])

        embeddings = data.get("embeddings", [])
        documents = data.get("documents", [])

        # Must check length, NOT "if embeddings"
        if embeddings is None or len(embeddings) == 0:
            continue

        for e, doc in zip(embeddings, documents):
            e = np.array(e).flatten().tolist()
            all_emb.append(e)
            all_labels.append(doc[:30])
            colors.append(palette[i % len(palette)])

    if len(all_emb) == 0:
        return "No embeddings available"

    X = np.vstack(all_emb)
    pts = PCA(n_components=2).fit_transform(X).tolist()

    return render_template(
        "visualize.html",
        points=pts,
        labels=all_labels,
        colors=colors
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
