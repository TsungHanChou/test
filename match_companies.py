import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Step 1: Load the Data ---
canonical_df = pd.read_csv('canonical_names.tsv', sep='\t')
messy_df = pd.read_csv('messy_company_names.tsv', sep='\t')

print("--- Data Loaded ---")
print("Canonical Names:")
print(canonical_df.head())
print("\nMessy Names:")
print(messy_df.head())

# --- Step 2: Initialize the AI Model ---
print("\n--- Initializing AI Model (this may take a moment on first run)... ---")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 3: Generate Vector Embeddings ---
print("--- Generating Vector Embeddings ---")
canonical_names = canonical_df['canonical_name'].tolist()
messy_names = messy_df['name_std'].tolist()

canonical_embeddings = model.encode(canonical_names, show_progress_bar=True)
messy_embeddings = model.encode(messy_names, show_progress_bar=True)

print(f"Shape of canonical embeddings: {canonical_embeddings.shape}")
print(f"Shape of messy embeddings: {messy_embeddings.shape}")

# --- Step 4: Calculate Similarity and Find Best Match ---
print("\n--- Calculating Similarities and Finding Best Matches ---")
similarity_matrix = cosine_similarity(messy_embeddings, canonical_embeddings)

best_match_indices = np.argmax(similarity_matrix, axis=1)
best_match_scores = np.max(similarity_matrix, axis=1)

# --- Step 5: Create the Final Results DataFrame ---
results_df = messy_df.copy()
results_df['best_canonical_match'] = canonical_df['canonical_name'].iloc[best_match_indices].values
results_df['ai_similarity_score'] = best_match_scores

print("\n--- Final Matching Results ---")
print(results_df[['id_name', 'name_std', 'best_canonical_match', 'ai_similarity_score']])

print("\n--- Potential Mismatches (Score < 0.85) ---")
low_confidence_matches = results_df[results_df['ai_similarity_score'] < 0.85]
print(low_confidence_matches[['name_std', 'best_canonical_match', 'ai_similarity_score']])
