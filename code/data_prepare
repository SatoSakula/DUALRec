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
