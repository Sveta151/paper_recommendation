import json
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Loading model
print("Loading model...")
model = SentenceTransformer("intfloat/e5-large-v2")  

# Loading data
with open("iclr2025_submissions.json", "r", encoding="utf-8") as f:
    papers_full = json.load(f)

with open("iclr2025_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    papers_embeds = data['embeddings']

# FAISS index for search 
dimension = papers_embeds.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(papers_embeds))


# Search function
def search(query, top_k=10, spotlight_only=False):
    query_embedding = model.encode("query: " + query, normalize_embeddings=True)
    #overfetching for spotlight cases
    D, I = index.search(np.array([query_embedding]), k=top_k * 2) 
    results = [papers_full[i] for i in I[0] if i < len(papers_full)]
    # in case want spotlight only papers
    if spotlight_only:
        results = [r for r in results if 'spotlight' in r.get('venue', '').lower()]
    return results[:top_k]


def print_results(results, show_abstract=False):
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} ({res['venue']})")
        print(f"   Authors: {', '.join(res['authors'])}")
        print(f"   Paper: {res['paper_url']}")
        if show_abstract:
            print("\n   Abstract:")
            print(f"   {res.get('abstract', 'No abstract available.')}")
        print("=" * 60)




# -------- Query Loop --------
print("Type your query and press Enter.")
print("Use the following optional flags:")
print("  --abstract     to show abstracts")
print("  --spotlight    to show only spotlight papers")
print("  --number=15    to control how many results are shown (default = 10)")
print("Type 'exit' or 'quit' to end the session.")

while True:
    user_input = input("\n Query: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Done")
        break

    # Parse flags
    show_abstract = "--abstract" in user_input
    spotlight_only = "--spotlight" in user_input

    # Extract --number=X (default to 10)
    number_match = re.search(r"--number=(\d+)", user_input)
    top_k = int(number_match.group(1)) if number_match else 10

    # Clean the query string
    cleaned_query = (
        user_input.replace("--abstract", "")
                  .replace("--spotlight", "")
                  .replace(number_match.group(0), "") if number_match else user_input
    ).strip()

    if not cleaned_query:
        print("Please enter a query (not just flags).")
        continue

    results = search(cleaned_query, top_k=top_k, spotlight_only=spotlight_only)
    print_results(results, show_abstract=show_abstract)