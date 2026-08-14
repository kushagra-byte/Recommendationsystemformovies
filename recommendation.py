import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random

# ==================================================
# Load Dataset
# ==================================================
print("Loading dataset...")
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print("Dataset loaded successfully!")
print(f"Total ratings : {len(ratings)}")
print(f"Total users   : {ratings['userId'].nunique()}")
print(f"Total movies  : {ratings['movieId'].nunique()}\n")

# ==================================================
# Preprocessing
# ==================================================
user_ids = ratings["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_ids = ratings["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

ratings["user"] = ratings["userId"].map(user2user_encoded)
ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

min_rating = ratings["rating"].min()
max_rating = ratings["rating"].max()
ratings["rating_norm"] = (ratings["rating"] - min_rating) / (max_rating - min_rating)

x = ratings[["user", "movie"]].values
y = ratings["rating_norm"].values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# ==================================================
# Model
# ==================================================
EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies, embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        dot = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot + user_bias + movie_bias
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.RootMeanSquaredError()]
)

print("Training the model... Please wait")
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=512,          # Increased for speed
    epochs=5,                # Reduced for speed
    verbose=1,
    validation_data=(x_val, y_val)
)

# ==================================================
# Recommendation Function
# ==================================================
def get_recommendations(user_id, model, ratings_df, movies_df, n=10, diversity=0.10):
    if user_id not in user2user_encoded:
        return []

    user_encoded = user2user_encoded[user_id]
    watched = set(ratings_df[ratings_df.userId == user_id]["movieId"].values)

    candidates = [m for m in movies_df["movieId"].unique()
                  if m not in watched and m in movie2movie_encoded]

    user_movie_array = np.hstack((
        [[user_encoded]] * len(candidates),
        [[movie2movie_encoded[m]] for m in candidates]
    ))

    preds = model.predict(user_movie_array, verbose=0).flatten()

    # Add slight randomness
    noisy_preds = preds + np.random.uniform(-diversity, diversity, size=len(preds))
    top_indices = noisy_preds.argsort()[-n:][::-1]

    results = []
    for idx in top_indices:
        movie_id = candidates[idx]
        title = movies_df[movies_df.movieId == movie_id]["title"].values[0]
        score = float(preds[idx])
        results.append((title, score))

    return results


# ==================================================
# MAIN
# ==================================================
available_users = ratings['userId'].unique().tolist()
USER_ID = random.choice(available_users)

print("\n" + "=" * 60)
print("RANDOM USER SELECTED → Anonymous User")
print("=" * 60)

print(f"\nTop 10 Recommendations for this user:")
print("-" * 50)

recommendations = get_recommendations(USER_ID, model, ratings, movies, n=10)

for i, (title, score) in enumerate(recommendations, 1):
    print(f"{i:2d}. {title}")
    print(f"     Predicted Score: {score:.3f}")