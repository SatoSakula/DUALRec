import os
import json
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# ---------------- CONFIG ---------------- #
CACHE_PATH = "deepseek_cache.json"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        deepseek_cache = json.load(f)
else:
    deepseek_cache = {}

model_emb = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- API CALL ---------------- #
def call_deepseek_via_openrouter(prompt, api_key):
    if prompt in deepseek_cache:
        return deepseek_cache[prompt]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "",
        "X-Title": "LSTM LLM Evaluation"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9
    }
    response = requests.post（）, headers=headers, json=data)
    result = response.json()
    if response.status_code == 200 and "choices" in result:
        content = result["choices"][0]["message"]["content"].strip()
        deepseek_cache[prompt] = content
        with open(CACHE_PATH, "w") as f:
            json.dump(deepseek_cache, f)
        return content
    else:
        print("❌ API Error:", result)
        return ""

# ---------------- PROMPT GENERATOR ---------------- #
def generate_deepseek_prompt(user_movie_ids, movies_df, lstm_recommendation):
    titles = []
    for mid in user_movie_ids:
        row = movies_df[movies_df['movieId'] == mid].iloc[0]
        titles.append(f"{row['title']} ({row['genres'].replace('|', ', ')})")
    history_str = "\n- " + "\n- ".join(titles)

    prompt = f"""
Below is a list of movies the user has watched. They enjoy similar themes, genres, or storytelling styles.
{history_str}

The system has identified the movie: {lstm_recommendation} as a good next suggestion.

Now, please recommend 10 movie titles the user would enjoy next.
Only return full movie titles with release years in the format: Title (Year).
These should align in style with the above list AND follow the direction of the LSTM recommendation.
Only output titles, one per line, no explanation.

Recommendations:
"""
    return prompt

# ---------------- HELPERS ---------------- #
def extract_recommendations(text):
    lines = text.split("\n")
    recommendations = []
    for line in lines:
        clean = re.sub(r"^[\*\d\.\s]*", "", line.strip())
        match = re.search(r'([^\(]+)\((\d{4})\)', clean)
        if match:
            recommendations.append(match.group(0).strip())
    return recommendations

def genre_jaccard_similarity(genres_a, genres_b):
    set_a = set(genres_a.split('|'))
    set_b = set(genres_b.split('|'))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0

def multi_ground_truth_hit(recommended, ground_truth_titles, k, threshold=0.9):
    for i, rec in enumerate(recommended[:k]):
        for gt in ground_truth_titles:
            score = SequenceMatcher(None, rec.lower(), gt.lower()).ratio()
            if score >= threshold:
                return 1, 1 / np.log2(i + 2)
    return 0, 0

def rerank_by_embedding(recommended_titles, reference_titles, top_k=10, verbose=False):
    all_texts = recommended_titles + reference_titles
    embeddings = model_emb.encode(all_texts, convert_to_tensor=True)
    rec_embeds = embeddings[:len(recommended_titles)]
    ref_embeds = embeddings[len(recommended_titles):]
    scores = util.cos_sim(rec_embeds, ref_embeds).max(dim=1).values
    scored_recs = list(zip(recommended_titles, scores.tolist()))
    ranked = sorted(scored_recs, key=lambda x: x[1], reverse=True)
    if verbose:
        print("\n🔁 Reranked by Sentence-BERT similarity:")
        for title, score in ranked:
            print(f"{title} (sim={score:.4f})")
    return [x[0] for x in ranked[:top_k]]
