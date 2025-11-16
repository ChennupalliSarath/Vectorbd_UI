Vector Database UI — Flask + Chroma + HuggingFace

A simple yet powerful web-based interface to manage, search, and visualize vector embeddings.
Built with Flask, ChromaDB, and Sentence Transformers (HuggingFace).

This app provides a complete workflow — from uploading text or files, generating embeddings, storing them, searching, and visualizing.
Features
* Collection Management
  Create, list, delete, and export collections.
  View collection summaries and recent additions.
* Data Upload
  Upload plain text or files (.txt, .pdf, .docx).
  Automatic embedding generation using HuggingFace models.
* Hybrid Search
  Choose between Vector, Keyword, or Hybrid search.
  Optional metadata-based filtering using JSON format.
* Visualization
  Interactive PCA-based 2D scatter plot of embeddings using Plotly.js.

* Dashboard
  Quick action cards, collection summary, and recently used collections.

* Export and Delete
  Export collections to JSON and delete specific documents or collections.

Tech Stack
Backend: Flask (Python)
Database: ChromaDB (Persistent Client)
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Frontend: HTML, CSS, JavaScript
Visualization: Plotly.js (local version)
File Processing: PyMuPDF (fitz), python-docx
Styling: Custom CSS

Project Structure

vector-db-ui/
│
├── app.py
├── requirements.txt
├── chroma_db/              # Chroma storage (auto-created)
├── uploads/                # Uploaded files
├── static/
│   ├── style.css
│   └── plotly.min.js
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── collections.html
│   ├── upload.html
│   ├── search.html
│   └── visualize.html
└── README.md
 Installation and Setup

1. Clone the repository

git clone https://github.com/<your-username>/vector-db-ui.git
cd vector-db-ui

2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac
3. Install dependencies
pip install -r requirements.txt
Example requirements.txt
flask
chromadb
sentence-transformers
scikit-learn
numpy
pymupdf
python-docx
4. Run the app

python app.py
Open in browser:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

 Usage Guide
* Manage Collections
  Create, list, delete, and export collections.

* Upload Data
  Upload text or files (.txt, .pdf, .docx).
  Optional metadata can be added in JSON format.
  Embeddings are generated automatically.

* Search
  Choose Vector, Keyword, or Hybrid mode.
  Optionally apply JSON-based metadata filters.

* Visualization
  Explore embeddings interactively in 2D PCA plot.

 Local Plotly Setup

If offline or for faster load times:

1. Download plotly.min.js from
   [https://cdn.plot.ly/plotly-latest.min.js](https://cdn.plot.ly/plotly-latest.min.js)
2. Save it inside the static folder as static/plotly.min.js.
   The app automatically uses this local file.

Export Format Example

Example JSON file generated when exporting a collection:

[
  {
    "document": "AI is transforming industries through automation.",
    "metadata": {"topic": "AI", "timestamp": "2025-11-17 14:25:01"},
    "embedding": [0.021, 0.88, ...]
  },
  {
    "document": "Machine learning improves model accuracy over time.",
    "metadata": {"topic": "ML", "timestamp": "2025-11-17 14:26:09"},
    "embedding": [0.12, 0.94, ...]
  }
]

UI Overview

Home Dashboard

* Quick action cards at the top.
* Summary of total collections and documents.
* Recently used collections with export buttons.

Upload Page

* Option to upload text or files.
* Metadata field for JSON input.
* Generates and stores embeddings.

Search Page

* Hybrid and vector-based search modes.
* Filter results using metadata.
* Delete documents directly from search results.

Visualization Page

* Displays embeddings using interactive 2D PCA scatter plot.

 Future Enhancements

* Add user authentication (Flask-Login or JWT).
* Chunk large documents automatically.
* Enable multi-file upload.
* Use UMAP or t-SNE for 3D visualization.
* Deploy on Render, Railway, or AWS.

License

This project is licensed under the MIT License.

Acknowledgements

* ChromaDB - Vector database engine
* HuggingFace Sentence Transformers - Embedding models
* Plotly.js - Visualization library
