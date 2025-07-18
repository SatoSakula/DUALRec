from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm

# 加载模型
deepseek_tokenizer = AutoTokenizer.from_pretrained(" ", trust_remote_code=True)
deepseek_model = AutoModelForCausalLM.from_pretrained(" ", trust_remote_code=True).to("cuda")
deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token

# 生成推荐
def generate_deepseek_recommendations(prompt, max_tokens=256):
    input_ids = deepseek_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = deepseek_model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
    return deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True).split("###")[-1].strip()

# 评估函数
def evaluate_deepseek_model(test_data, movies_df, movie2idx, title_features, genre_features, lstm_model, k_values=[1, 5], user_sample_size=150, print_first_n=3):
    model_emb = SentenceTransformer("all-MiniLM-L6-v2")
    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    genre_similarities = []
    all_users = test_data['userId'].unique()
    sampled_users = np.random.choice(all_users, size=min(user_sample_size, len(all_users)), replace=False)
    printed = 0

    def extract_recommendations(text):
        lines = text.strip().split("\n")
        recommendations = []
        for line in lines:
            clean = re.sub(r"^[\*\d\.\s]*", "", line.strip())
            match = re.search(r'([^\(]+)\((\d{4})\)', clean)
            if match:
                recommendations.append(match.group(0).strip())
        return recommendations[:10]

    def rerank_by_embedding(recommended_titles, reference_titles, top_k=10):
        if not recommended_titles:
            return []
        all_texts = recommended_titles + reference_titles
        embeddings = model_emb.encode(all_texts, convert_to_tensor=True)
        rec_embeds = embeddings[:len(recommended_titles)]
        ref_embeds = embeddings[len(recommended_titles):]
        scores = util.cos_sim(rec_embeds, ref_embeds).max(dim=1).values
        scored_recs = list(zip(recommended_titles, scores.tolist()))
        ranked = sorted(scored_recs, key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked[:top_k]]

    def multi_ground_truth_hit(recommended, ground_truth_titles, k, threshold=0.8):
        for i, rec in enumerate(recommended[:k]):
            for gt in ground_truth_titles:
                score = SequenceMatcher(None, rec.lower(), gt.lower()).ratio()
                if score >= threshold:
                    return 1, 1 / np.log2(i + 2)
        return 0, 0

    def genre_jaccard_similarity(genres_a, genres_b):
        set_a = set(genres_a.split('|'))
        set_b = set(genres_b.split('|'))
        return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0

    def generate_prompt(user_movie_ids, movies_df, lstm_rec):
        titles = []
        for mid in user_movie_ids[-3:]:
            try:
                row = movies_df[movies_df['movieId'] == mid].iloc[0]
                titles.append(row['title'])
            except:
                continue
        history_str = ", ".join(titles)
        return f"""Given the user's watched movies and LSTM recommendation, generate 3 more movies the user will likely enjoy.
Watched: {history_str}
LSTM Suggests: {lstm_rec}
###"""

    for user_id in tqdm(sampled_users, desc="Evaluating users"):
        user_data = test_data[test_data['userId'] == user_id].sort_values('timestamp')
        if len(user_data) < 10:
            continue

        movie_ids = user_data['movieId'].values
        input_history = movie_ids[:-5]
        ground_truth_ids = movie_ids[-5:]
        ground_truth_titles = [movies_df[movies_df['movieId'] == mid].iloc[0]['title'] for mid in ground_truth_ids]

        try:
            lstm_rec = get_lstm_based_recommendation(
                model=lstm_model,
                user_movie_ids=input_history,
                title_data=title_features,
                genre_data=genre_features,
                movie2idx=movie2idx,
                idx2movie={v: k for k, v in movie2idx.items()},
                movies_df=movies_df
            )

            prompt = generate_prompt(input_history, movies_df, lstm_rec)
            llm_response = generate_deepseek_recommendations(prompt)
            recommended = extract_recommendations(llm_response)

            if recommended:
                reference = [lstm_rec] + ground_truth_titles
                recommended = rerank_by_embedding(recommended, reference, top_k=max(k_values))

            if printed < print_first_n:
                print(f"\n🧪 User {user_id}")
                print("GT:", ground_truth_titles)
                print("LSTM:", lstm_rec)
                print("DeepSeek:", recommended)
                printed += 1

            for k in k_values:
                hr, ndcg = multi_ground_truth_hit(recommended, ground_truth_titles, k)
                results[f"HR@{k}"].append(hr)
                results[f"NDCG@{k}"].append(ndcg)

            # Genre similarity
            if recommended:
                top1 = recommended[0]
                best_match = None
                best_score = 0
                for _, movie_row in movies_df.iterrows():
                    score = SequenceMatcher(None, top1.lower(), movie_row['title'].lower()).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = movie_row

                if best_match is not None and best_score > 0.6:
                    rec_genres = best_match['genres']
                    gt_similarities = [
                        genre_jaccard_similarity(movies_df[movies_df['movieId'] == gt_id].iloc[0]['genres'], rec_genres)
                        for gt_id in ground_truth_ids
                    ]
                    genre_similarities.append(max(gt_similarities))

        except Exception as e:
            print(f"⚠️ User {user_id} error: {e}")
            continue

    print(f"\n📊 Evaluation Summary:")
    for k in k_values:
        print(f"HR@{k}: {np.mean(results[f'HR@{k}']):.4f} | NDCG@{k}: {np.mean(results[f'NDCG@{k}']):.4f}")
    if genre_similarities:
        print(f"🎭 Genre similarity (Top-1 vs any GT): {np.mean(genre_similarities):.4f}")
