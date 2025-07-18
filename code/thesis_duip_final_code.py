# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Load the datasets
movies = pd.read_csv( )
ratings = pd.read_csv()  

# Merge the datasets on 'movieId'
data = pd.merge(ratings, movies, on='movieId')

# Create a mapping from movieId to unique indices
top_movies = data['movieId'].value_counts().head(1000).index
data = data[data['movieId'].isin(top_movies)]
movie2idx = {movie: idx for idx, movie in enumerate(data['movieId'].unique())}
data['movie_idx'] = data['movieId'].map(movie2idx)

# Split data into train, validation, and test sets
# Example split: 70% train, 15% validation, 15% test
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Tokenize Movie Titles
title_tokenizer = Tokenizer(num_words=5000, lower=True)
title_tokenizer.fit_on_texts(movies['title'])

# Convert titles to sequences
title_sequences = title_tokenizer.texts_to_sequences(movies['title'])
max_title_length = 10  # Maximum words per title
title_padded = pad_sequences(title_sequences, maxlen=max_title_length, padding='post')

# Create title lookup dictionary
title_features = {}
for idx, row in movies.iterrows():
    title_features[row['movieId']] = title_padded[idx]

# 2. Process Genres (Multi-hot encoding)
movies['genre_list'] = movies['genres'].str.split('|')
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genre_list'])
genre_names = mlb.classes_
n_genres = len(genre_names)

# Create genre lookup dictionary
genre_features = {}
for idx, row in movies.iterrows():
    genre_features[row['movieId']] = genre_encoded[idx]

print(f"Number of unique genre categories: {n_genres}")
print(f"Genre categories: {genre_names}")
print(f"Title vocabulary size: {len(title_tokenizer.word_index)}")

import torch
from torch.utils.data import Dataset, DataLoader

# Function to generate sequences for training
class MovieSequenceDataset:
    def __init__(self, df, seq_len=30, title_features=None, genre_features=None):
        self.sequences = []
        self.title_sequences = []
        self.genre_sequences = []
        self.targets = []
        grouped = df.groupby('userId')

        for user_id, user_data in grouped:
            user_data = user_data.sort_values('timestamp')
            movie_ids = user_data['movieId'].values
            movie_indices = user_data['movie_idx'].values

            if len(movie_indices) > seq_len:
                for i in range(len(movie_indices) - seq_len):
                    # Movie ID sequence
                    seq_input = movie_indices[i:i+seq_len]
                    seq_target = movie_indices[i+seq_len]

                    # Title features for the sequence
                    title_seq = np.zeros((seq_len, max_title_length))
                    genre_seq = np.zeros((seq_len, n_genres))

                    for j, movie_id in enumerate(movie_ids[i:i+seq_len]):
                        if title_features and movie_id in title_features:
                            title_seq[j] = title_features[movie_id]
                        if genre_features and movie_id in genre_features:
                            genre_seq[j] = genre_features[movie_id]

                    self.sequences.append(seq_input)
                    self.title_sequences.append(title_seq)
                    self.genre_sequences.append(genre_seq)
                    self.targets.append(seq_target)

        # Convert to numpy arrays
        self.sequences = np.array(self.sequences)
        self.title_sequences = np.array(self.title_sequences)
        self.genre_sequences = np.array(self.genre_sequences)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'movie_ids': self.sequences[idx],
            'titles': self.title_sequences[idx],
            'genres': self.genre_sequences[idx],
            'target': self.targets[idx]
        }


# Create datasets
train_dataset = MovieSequenceDataset(
    train_data,
    seq_len=30,
    title_features=title_features,
    genre_features=genre_features
)

val_dataset = MovieSequenceDataset(
    val_data,
    seq_len=30,
    title_features=title_features,
    genre_features=genre_features
)

# Extract data for training
train_movie_ids = train_dataset.sequences
train_titles = train_dataset.title_sequences
train_genres = train_dataset.genre_sequences
train_y = train_dataset.targets

val_movie_ids = val_dataset.sequences
val_titles = val_dataset.title_sequences
val_genres = val_dataset.genre_sequences
val_y = val_dataset.targets

import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """
    Plot all training metrics from the history object
    """
    # Get all available metrics from history
    metrics = []
    for key in history.history.keys():
        if not key.startswith('val_'):
            metrics.append(key)

    # Create subplots - 2 rows x 3 columns for all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training History', fontsize=16)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot each metric
    plot_idx = 0
    for metric in metrics:
        # Training metric
        axes[plot_idx].plot(history.history[metric], label=f'Train {metric}')

        # Validation metric if it exists
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            axes[plot_idx].plot(history.history[val_metric], label=f'Val {metric}')

        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel(metric.replace('_', ' ').title())
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_title(f'{metric.replace("_", " ").title()} History')

        plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Create detailed individual plots for better visualization
    create_detailed_plots(history)

def create_detailed_plots(history):
    """
    Create detailed individual plots for each metric
    """
    # Plot 1: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Top-K Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['sparse_top_k_categorical_accuracy'],
             label='Training Top-5 Accuracy', linewidth=2)
    plt.plot(history.history['val_sparse_top_k_categorical_accuracy'],
             label='Validation Top-5 Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Top-5 Accuracy', fontsize=12)
    plt.title('Top-5 Accuracy History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 4: Combined view with dual y-axis
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot loss on the first y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(history.history['loss'], color=color, label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], color='lightcoral', label='Val Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)
    ax2.plot(history.history['accuracy'], color=color, label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], color='lightblue', label='Val Accuracy', linewidth=2)
    ax2.plot(history.history['sparse_top_k_categorical_accuracy'],
             color='darkblue', label='Train Top-5', linewidth=2, linestyle='--')
    ax2.plot(history.history['val_sparse_top_k_categorical_accuracy'],
             color='royalblue', label='Val Top-5', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

    plt.title('Combined Training Metrics', fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_improvement_over_time(history):
    """
    Plot improvement percentages from initial values
    """
    metrics_to_plot = ['accuracy', 'sparse_top_k_categorical_accuracy']

    plt.figure(figsize=(12, 8))

    for metric in metrics_to_plot:
        if metric in history.history:
            # Training improvement
            train_values = np.array(history.history[metric])
            train_improvement = (train_values - train_values[0]) / (train_values[0] + 1e-10) * 100
            plt.plot(train_improvement, label=f'Train {metric} improvement (%)', linewidth=2)

            # Validation improvement
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                val_values = np.array(history.history[val_metric])
                val_improvement = (val_values - val_values[0]) / (val_values[0] + 1e-10) * 100
                plt.plot(val_improvement, label=f'Val {metric} improvement (%)', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title('Relative Improvement from Initial Values', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_summary_table(history):
    """
    Create a summary table of final metrics
    """
    final_epoch = len(history.history['loss']) - 1

    print("\n" + "="*60)
    print("TRAINING SUMMARY - Final Metrics (Epoch {})".format(final_epoch + 1))
    print("="*60)

    metrics_display = {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
        'sparse_top_k_categorical_accuracy': 'Top-5 Accuracy'
    }

    for metric_key, metric_name in metrics_display.items():
        if metric_key in history.history:
            train_val = history.history[metric_key][final_epoch]
            val_key = f'val_{metric_key}'

            if val_key in history.history:
                val_val = history.history[val_key][final_epoch]

                # Format based on whether it's loss or accuracy
                if 'loss' in metric_key:
                    print(f"{metric_name:<20} | Train: {train_val:8.4f} | Val: {val_val:8.4f}")
                else:
                    print(f"{metric_name:<20} | Train: {train_val:8.2%} | Val: {val_val:8.2%}")

    print("="*60)

    # Best epoch for each metric
    print("\nBest Epochs:")
    print("-"*60)

    for metric_key, metric_name in metrics_display.items():
        val_key = f'val_{metric_key}'
        if val_key in history.history:
            if 'loss' in metric_key:
                best_epoch = np.argmin(history.history[val_key])
                best_value = np.min(history.history[val_key])
                print(f"{metric_name:<20} | Epoch: {best_epoch + 1:3d} | Value: {best_value:8.4f}")
            else:
                best_epoch = np.argmax(history.history[val_key])
                best_value = np.max(history.history[val_key])
                print(f"{metric_name:<20} | Epoch: {best_epoch + 1:3d} | Value: {best_value:8.2%}")

    print("="*60)

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Updated parameters based on actual data shapes
seq_length = 30
max_title_length = 10
n_genres = 18
vocab_size = 1000
title_vocab_size = 5000


def create_fixed_enhanced_model(seq_length=30, return_lstm_output=False):
    movie_input = Input(shape=(seq_length,), name='movie_sequence')
    title_input = Input(shape=(seq_length, max_title_length), name='title_sequence')
    genre_input = Input(shape=(seq_length, n_genres), name='genre_features')

    # Embeddings and processing (same as before)
    movie_embedded = Embedding(vocab_size, 128, name="movie_embedding")(movie_input)

    title_embedded_list = []
    for i in range(seq_length):
        title_slice = tf.keras.layers.Lambda(lambda x: x[:, i, :])(title_input)
        title_emb = Embedding(title_vocab_size, 64)(title_slice)
        title_pooled = GlobalAveragePooling1D()(title_emb)
        title_embedded_list.append(title_pooled)

    title_embedded = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(title_embedded_list)
    genre_dense = Dense(64, activation='relu')(genre_input)

    combined_features = Concatenate()([movie_embedded, title_embedded, genre_dense])

    # LSTM layers
    x = LSTM(256, return_sequences=True, dropout=0.3)(combined_features)
    lstm_output = LSTM(128, dropout=0.3)(x)

    if return_lstm_output:
        return Model(inputs=[movie_input, title_input, genre_input], outputs=lstm_output)

    # Otherwise, continue to final output
    x = Dense(256, activation='relu')(lstm_output)
    x = Dropout(0.3)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    return Model(inputs=[movie_input, title_input, genre_input], outputs=output)



# Data type fix for titles
def fix_title_dtype(train_titles, val_titles):
    """Convert title data to int32 as expected by Embedding layer"""
    train_titles_fixed = train_titles.astype(np.int32)
    val_titles_fixed = val_titles.astype(np.int32)
    return train_titles_fixed, val_titles_fixed

# Fix data types
train_titles_fixed, val_titles_fixed = fix_title_dtype(train_titles, val_titles)

# Create model
model = create_fixed_enhanced_model(seq_length=30)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
)

# Train with fixed data
history = model.fit(
    [train_movie_ids, train_titles_fixed, train_genres],
    train_y,
    batch_size=128,
    epochs=10,
    validation_data=([val_movie_ids, val_titles_fixed, val_genres], val_y)
)

plot_training_history(history)
plot_improvement_over_time(history)
create_summary_table(history)

def save_lstm_weights(model, save_path):
    """
    Extract and save LSTM layer weights from the model

    Args:
        model: The trained Keras model
        save_path: Directory to save the weights
    """
    import os

    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Find LSTM layers in the model
    lstm_layers = []
    for layer in model.layers:
        if 'lstm' in layer.name.lower():
            lstm_layers.append(layer)

    print(f"Found {len(lstm_layers)} LSTM layers in the model")

    # Save weights for each LSTM layer
    for i, lstm_layer in enumerate(lstm_layers):
        # Get weights
        weights = lstm_layer.get_weights()

        # Save weights
        weights_path = os.path.join(save_path, f"lstm_{i+1}_weights.npz")
        np.savez(weights_path, *weights)

        # Print layer info
        print(f"Saved weights for {lstm_layer.name} to {weights_path}")
        print(f"  - Shape of weights: {[w.shape for w in weights]}")

    # Save model diagram
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=os.path.join(save_path, 'model_diagram.png'), show_shapes=True)
        print(f"Saved model diagram to {os.path.join(save_path, 'model_diagram.png')}")
    except Exception as e:
        print(f"Could not save model diagram: {e}")

    # Also save complete model for reference
    model.save(os.path.join(save_path, 'full_model.h5'))
    print(f"Saved complete model to {os.path.join(save_path, 'full_model.h5')}")

    return lstm_layers

# Save LSTM weights
print("\nSaving LSTM weights...")
save_path = '/content/drive/MyDrive/Colab Notebooks/lstm_weights'
# Save the LSTM model weights
lstm_layers = save_lstm_weights(model=create_fixed_enhanced_model(seq_length=30), save_path=save_path)

"""# Hybrid recommendation system that combines the strengths of both the LSTM recommendation and LLM API calls"""

import requests

# Step 1: 从 LSTM 模型生成用户推荐
def get_lstm_based_recommendation(model, user_movie_ids, title_data, genre_data, movie2idx, idx2movie, movies_df):
    """
    使用 LSTM 模型为用户生成推荐电影（使用模型softmax输出argmax）
    """
    import numpy as np

    # 准备模型输入
    movie_seq = np.array([movie2idx[mid] for mid in user_movie_ids[-30:]])  # 保留序列长度为30
    movie_seq = np.pad(movie_seq, (30 - len(movie_seq), 0), mode='constant')  # Padding if needed

    title_seq = np.array([title_data[mid] for mid in user_movie_ids[-30:]])
    title_seq = np.pad(title_seq, ((30 - len(title_seq), 0), (0, 0)), mode='constant')

    genre_seq = np.array([genre_data[mid] for mid in user_movie_ids[-30:]])
    genre_seq = np.pad(genre_seq, ((30 - len(genre_seq), 0), (0, 0)), mode='constant')

    # 扩展维度给模型 (batch_size=1)
    inputs = [
        np.expand_dims(movie_seq, axis=0),
        np.expand_dims(title_seq, axis=0),
        np.expand_dims(genre_seq, axis=0)
    ]

    # 模型预测
    prediction = model.predict(inputs)
    predicted_idx = np.argmax(prediction, axis=1)[0]

    # 找到 movieId
    predicted_movie_id = [k for k, v in movie2idx.items() if v == predicted_idx][0]
    predicted_title = movies_df[movies_df['movieId'] == predicted_movie_id].iloc[0]['title']
    return predicted_title

# Step 2: 构造自然语言 Prompt（用户历史 + LSTM 推荐）
def generate_deepseek_prompt(user_movie_ids, movies_df, lstm_recommendation):
    """
    生成自然语言提示语，用于发送给 LLM
    """
    history_titles = []
    for mid in user_movie_ids:
        row = movies_df[movies_df['movieId'] == mid].iloc[0]
        history_titles.append(f"{row['title']} ({row['genres'].replace('|', ', ')})")

    history_str = "\n- " + "\n- ".join(history_titles)
    prompt = f"""Below is a user's movie watching history:{history_str}

Based on this, the system (LSTM) recommends: {lstm_recommendation}.

Now, as a helpful assistant, recommend 3 more full movie titles with release years that this user would likely enjoy next.
Recommendations:"""
    return prompt

# Step 3: 通过 DeepSeek API 调用
def call_deepseek_via_openrouter(prompt, api_key):
    """
    使用 DeepSeek Chat API 生成推荐
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer":
        "X-Title": "LSTM to DeepSeek Rec"
    }

    data = {
        "model": "deepseek/deepseek-chat",  # 已知可用的模型
        "messages": [
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post("", headers=headers, json=data)
    result = response.json()

    if response.status_code == 200 and "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    else:
        print("Error from API:", result)
        return "No valid response received."

# Step 4: 示例运行整合流程
def full_lstm_llm_pipeline():
    sample_user_id = train_data['userId'].unique()[0]
    user_data = train_data[train_data['userId'] == sample_user_id].sort_values('timestamp')
    user_movie_ids = user_data['movieId'].values[-5:]  # 使用最近5部

    lstm_rec = get_lstm_based_recommendation(
        model=model,
        user_movie_ids=user_movie_ids,
        title_data=title_features,
        genre_data=genre_features,
        movie2idx=movie2idx,
        idx2movie={v: k for k, v in movie2idx.items()},
        movies_df=movies
    )

    prompt = generate_deepseek_prompt(user_movie_ids, movies, lstm_rec)

    print("\n📝 Prompt to LLM:\n")
    print(prompt)

    deepseek_api_key =
    response = call_deepseek_via_openrouter(prompt, deepseek_api_key)

    print("\n🎬 Final LLM Recommendations:\n")
    print(response)

# ✨ 执行推荐生成流程
full_lstm_llm_pipeline()

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

# ---------------- MAIN EVALUATION ---------------- #
def evaluate_model_on_sampled_users(test_data, movies_df, movie2idx, title_features, genre_features,
                                    model, api_key, k_values=[1, 5], user_sample_size=150, print_first_n=5):

    all_users = test_data['userId'].unique()
    sampled_users = np.random.choice(all_users, size=min(user_sample_size, len(all_users)), replace=False)

    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    genre_similarities = []

    printed = 0

    for user_id in tqdm(sampled_users, desc="Evaluating users"):
        user_data = test_data[test_data['userId'] == user_id].sort_values('timestamp')
        if len(user_data) < 10:
            continue

        movie_ids = user_data['movieId'].values
        input_history = movie_ids[:-5]
        ground_truth_ids = movie_ids[-5:]
        ground_truth_titles = [movies_df[movies_df['movieId'] == mid].iloc[0]['title'] for mid in ground_truth_ids]

        try:
            # 📌 Record LSTM baseline top-1 recommendation
            lstm_title = get_lstm_based_recommendation(
                model=model,
                user_movie_ids=input_history,
                title_data=title_features,
                genre_data=genre_features,
                movie2idx=movie2idx,
                idx2movie={v: k for k, v in movie2idx.items()},
                movies_df=movies_df
            )

            prompt = generate_deepseek_prompt(input_history, movies_df, lstm_title)
            llm_response = call_deepseek_via_openrouter(prompt, api_key)
            recommended = extract_recommendations(llm_response)
            if len(recommended) > 0:
                reference_titles = [lstm_title] + ground_truth_titles
                recommended = rerank_by_embedding(recommended, reference_titles, top_k=10, verbose=(printed < print_first_n))

            if printed < print_first_n:
                print(f"\n🧪 User {user_id}")
                print("Ground truth (5):", ground_truth_titles)
                print("LSTM predicted:", lstm_title)
                print("LLM Recommendations:")
                for i, r in enumerate(recommended):
                    print(f"{i+1}. {r}")
                printed += 1

            for k in k_values:
                hr, ndcg = multi_ground_truth_hit(recommended, ground_truth_titles, k)
                results[f"HR@{k}"].append(hr)
                results[f"NDCG@{k}"].append(ndcg)

            if recommended:
                top1 = recommended[0]
                matched = movies_df[movies_df['title'].str.lower() == top1.lower()]
                if not matched.empty:
                    rec_genres = matched.iloc[0]['genres']
                    gt_similarities = []
                    for gt_id in ground_truth_ids:
                        gt_genres = movies_df[movies_df['movieId'] == gt_id].iloc[0]['genres']
                        gt_similarities.append(genre_jaccard_similarity(gt_genres, rec_genres))
                    genre_similarities.append(max(gt_similarities))

        except Exception as e:
            print(f"⚠️ User {user_id} error: {e}")
            continue

    print("\n📊 Evaluation Results:")
    for k in k_values:
        print(f"HR@{k}: {np.mean(results[f'HR@{k}']):.4f} | NDCG@{k}: {np.mean(results[f'NDCG@{k}']):.4f}")
    print(f"🎭 Genre similarity (Top-1 vs any GT): {np.mean(genre_similarities) if genre_similarities else 0:.4f}")

    return results, genre_similarities

results, genre_sims = evaluate_model_on_sampled_users(
    test_data=test_data,
    movies_df=movies,
    movie2idx=movie2idx,
    title_features=title_features,
    genre_features=genre_features,
    model=model,
    api_key=,
    user_sample_size=928,
    print_first_n=5
)

# ✅ Generate training samples for LoRA fine-tuning using LSTM predictions
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType


def build_finetune_dataset_from_lstm(train_data, movies_df, model, title_features, genre_features, movie2idx, save_path):
    samples = []
    all_users = train_data['userId'].unique()
    for user_id in tqdm(all_users, desc="Building LoRA training samples"):
        user_data = train_data[train_data['userId'] == user_id].sort_values('timestamp')
        if len(user_data) < 10:
            continue
        movie_ids = user_data['movieId'].values
        input_history = movie_ids[:-5]
        target_ids = movie_ids[-5:]

        watched_titles = [
            movies_df[movies_df['movieId'] == mid].iloc[0]['title']
            for mid in input_history[-5:]
        ]
        history_text = ", ".join(watched_titles)

        try:
            lstm_title = get_lstm_based_recommendation(
                model=model,
                user_movie_ids=input_history,
                title_data=title_features,
                genre_data=genre_features,
                movie2idx=movie2idx,
                idx2movie={v: k for k, v in movie2idx.items()},
                movies_df=movies_df
            )
        except:
            continue

        target_titles = [
            movies_df[movies_df['movieId'] == mid].iloc[0]['title']
            for mid in target_ids[:3]
        ]
        rec_output = "\n".join([f"{i+1}. {t}" for i, t in enumerate(target_titles)])

        samples.append({
            "instruction": "Given the user's watched movies and LSTM recommendation, generate 3 more movies the user will likely enjoy.",
            "input": f"Watched: {history_text}\nLSTM Suggests: {lstm_title}",
            "output": rec_output
        })

    with open(save_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Saved {len(samples)} training samples to {save_path}")

build_finetune_dataset_from_lstm(
    train_data=train_data,
    movies_df=movies,
    model=model,
    title_features=title_features,
    genre_features=genre_features,
    movie2idx=movie2idx,
    save_path=
)

"""
# 基于推荐任务和用户历史，微调DeepSeek  VL2模型"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch, json

def finetune_deepseek_lora(jsonl_path, base_model_name="", output_dir=""):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    with open(jsonl_path) as f:
        data = [json.loads(line.strip()) for line in f]
    dataset = Dataset.from_list(data)

    def tokenize(example):
        prompt = f"{example['instruction']}\n{example['input']}\n###\n{example['output']}"
        tokens = tokenizer(prompt, padding='max_length', truncation=True, max_length=512)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Finetuned model saved to {output_dir}")

finetune_deepseek_lora(
    jsonl_path="",
    base_model_name="",
    output_dir=""
)

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

results, genre_sims = evaluate_deepseek_model(
    test_data=test_data,
    movies_df=movies,
    movie2idx=movie2idx,
    title_features=title_features,
    genre_features=genre_features,
    lstm_model=model,
    k_values=[1, 5],
    user_sample_size=928,
    print_first_n=3
)