# ICLR 2025 Paper Finder

This is a simple terminal-based tool for finding relevant papers from ICLR 2025 using semantic search.

It uses:
- `intfloat/e5-large-v2`, an open-source embedding model
- `faiss` for efficient vector similarity search
- Paper data extracted from the OpenReview API

## Getting Started

### 1. Clone the repository
### 2. Create and activate a virtual environment

```console
python -m venv env
source env/bin/activate 
```
### 3. Install dependencies

```console
pip install -r requirements.txt
```

### 4. Run the script 
```console
python find_papers.py
```

It may take some time to load the model the first time. Once loaded, you can enter queries in the terminal.

Features
	•	Displays title, authors, and a direct link to the paper.
	•	Optional flags:
	•	--abstract to show abstracts
	•	--spotlight to filter only spotlight papers
	•	--number=20 to control the number of returned results (default is 10)

**Example:**
`Query: multimodal models --abstract --spotlight --number=15`
