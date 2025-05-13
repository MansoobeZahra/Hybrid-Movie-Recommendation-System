# 🎬 Hybrid Movie Recommendation System

A powerful hybrid movie recommendation engine combining **Content-Based Filtering** and **Collaborative Filtering** techniques using **Python**, **Pandas**, **Scikit-learn**, and **Surprise** libraries.

---

## 📌 Project Overview

This system intelligently recommends movies to users by blending two core methods:

- **Content-Based Filtering**: Suggests movies similar to ones a user has liked, based on genres, tags, and descriptions.
- **Collaborative Filtering**: Predicts user preferences based on the ratings of other similar users.

Together, this hybrid approach improves recommendation accuracy by leveraging both item features and user behavior.

---

## 📁 Features

- 🔍 Recommends movies based on movie metadata (genres, overview, tags, keywords)
- 👥 Predicts ratings using user-user and item-item similarities
- 🧠 Combines **TF-IDF vectorization** and **cosine similarity** for content-based filtering
- 📊 Uses **Singular Value Decomposition (SVD)** from `surprise` for collaborative filtering
- 💡 Hybrid logic that blends both approaches for better personalization

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas**, **NumPy**
- **Scikit-learn** (TF-IDF, cosine similarity)
- **Surprise** (SVD algorithm)
- **Jupyter Notebook** for development and visualization

---

## 🧠 How It Works

### 🔹 Content-Based Filtering

- Uses movie metadata (genres, tags, plot) to create a TF-IDF matrix
- Calculates **cosine similarity** between movie vectors
- Recommends movies similar to the ones rated highly by the user

### 🔹 Collaborative Filtering

- Uses historical user ratings from the MovieLens dataset
- Trains a matrix factorization model using **SVD**
- Predicts unseen ratings based on similar users and items

### 🔹 Hybrid Logic

For a given movie and user:

1. Compute similar movies using content-based filtering  
2. Use collaborative filtering to estimate user ratings for those similar movies  
3. Recommend the top **N** movies with highest predicted ratings

---

## 📂 Dataset

- **MovieLens 100k Dataset**  
  📥 [Download Here](https://grouplens.org/datasets/movielens/100k)

---

## 🚀 Getting Started

### 1. Clone the Repo
 \
### 📷 Example Output
You liked The Matrix? Here are similar movies you might enjoy...

1. Inception (2010)
2. The Terminator (1984)
3. Minority Report (2002)
4. Interstellar (2014)
5. Edge of Tomorrow (2014)
---
###📈 Future Improvements

Add deep learning-based embeddings (e.g., autoencoders)

Integrate user profile input

Deploy with a Flask or Streamlit web interface

---

### 🙋‍♂️ Author
Mansoob-e-Zahra 12 May 2025
LinkedIn Mansoobezehra
Email  mansoobezehra@gmail.c[Movie_Recommnder.zip](https://github.com/user-attachments/files/20188654/Movie_Recommnder.zip)
om[Movie_Recommnder.zip](https://github.com/user-attachments/files/20188651/Movie_Recommnder.zip)

---

### ⭐️ Give it a Star!
If you find this useful, please ⭐️ the repository to support the project!

---



