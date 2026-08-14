import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.caption("Anonymous User • TensorFlow Collaborative Filtering")

# ==================================================
# Load Data + Train Model (Cached)
# ==================================================
@st.cache_resource
def load_and_train():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")

    user_ids = ratings["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = ratings["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    ratings = ratings.copy()
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
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    with st.spinner("Training model... (only happens once)"):
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=512,
            epochs=5,
            verbose=0,
            validation_data=(x_val, y_val)
        )

    return ratings, movies, model, user2user_encoded, movie2movie_encoded


ratings, movies, model, user2user_encoded, movie2movie_encoded = load_and_train()

# ==================================================
# Recommendation Function
# ==================================================
def get_recommendations(user_id, n=10, diversity=0.10):
    if user_id not in user2user_encoded:
        return []

    user_encoded = user2user_encoded[user_id]
    watched = set(ratings[ratings.userId == user_id]["movieId"].values)

    candidates = [m for m in movies["movieId"].unique()
                  if m not in watched and m in movie2movie_encoded]

    user_movie_array = np.hstack((
        [[user_encoded]] * len(candidates),
        [[movie2movie_encoded[m]] for m in candidates]
    ))

    preds = model.predict(user_movie_array, verbose=0).flatten()
    noisy_preds = preds + np.random.uniform(-diversity, diversity, size=len(preds))
    top_indices = noisy_preds.argsort()[-n:][::-1]

    results = []
    for idx in top_indices:
        movie_id = candidates[idx]
        title = movies[movies.movieId == movie_id]["title"].values[0]
        score = float(preds[idx])
        results.append((title, score))
    return results

# ==================================================
# Sidebar + Session State
# ==================================================
st.sidebar.header("Settings")

n_recs = st.sidebar.slider("Number of Recommendations", 5, 15, 10)

# Initialize session state properly
if "current_user" not in st.session_state:
    st.session_state.current_user = random.choice(list(user2user_encoded.keys()))

if st.sidebar.button("🎲 Get New Anonymous User"):
    st.session_state.current_user = random.choice(list(user2user_encoded.keys()))

# Always get the current user from session state
user_id = st.session_state.current_user

# ==================================================
# Main Content
# ==================================================
st.subheader("Anonymous User Recommendations")

with st.spinner("Generating recommendations..."):
    recommendations = get_recommendations(user_id, n=n_recs)

if not recommendations:
    st.warning("No recommendations found.")
else:
    for i, (title, score) in enumerate(recommendations, 1):
        st.markdown(f"**{i}. {title}**")
        st.progress(score, text=f"Predicted Score: {score:.3f}")

st.markdown("---")
st.caption("User identity is hidden • Model trained with TensorFlow")