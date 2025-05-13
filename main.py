import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load the movies dataset
# For this example, I'll use the MovieLens dataset format
# If you have a different dataset, adjust the column names accordingly
movies = pd.read_csv(r'C:\Users\Q No-804\Documents\MYPROJECTS\DataScience\Movie_Recommnder\movies.csv')
ratings = pd.read_csv(r'C:\Users\Q No-804\Documents\MYPROJECTS\DataScience\Movie_Recommnder\ratings.csv')

print("Movies dataset sample:")
print(movies.head())
print("\nRatings dataset sample:")
print(ratings.head())

# Data preprocessing
# Extract genres and create a combined feature for content-based filtering
def extract_features(movies_df):
    # Extract year from title
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
    
    # Clean the title by removing the year
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
    
    # Create a combined feature for content-based filtering
    movies_df['combined_features'] = movies_df['genres'].str.replace('|', ' ')
    
    return movies_df

movies = extract_features(movies)
print("\nPreprocessed movies:")
print(movies.head())

# Content-based recommendation system
def build_content_based_recommender(movies_df):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Apply TF-IDF to the combined features
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping of movie titles to indices
    indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()
    
    return cosine_sim, indices

# Build the recommender
cosine_sim, indices = build_content_based_recommender(movies)

# Function to get movie recommendations
def get_content_based_recommendations(title, cosine_sim=cosine_sim, indices=indices, movies_df=movies, num_recommendations=10):
    # Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        print(f"'{title}' not found in the dataset. Here are some available titles:")
        sample_titles = movies_df['clean_title'].sample(5).tolist()
        for t in sample_titles:
            print(f"- {t}")
        return None
    
    # Get the similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the most similar movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Return the top movies with their similarity scores
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['similarity_score'] = similarity_scores
    
    return recommendations[['title', 'genres', 'similarity_score']]

# Collaborative filtering recommendation
def build_collaborative_filtering_recommender(ratings_df, movies_df):
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Calculate movie-movie similarity matrix
    movie_similarity = cosine_similarity(user_item_matrix.T)
    
    # Create a mapping of movie IDs to indices
    movie_indices = {movie_id: i for i, movie_id in enumerate(user_item_matrix.columns)}
    
    # Create a mapping of indices to movie IDs
    indices_to_movieid = {i: movie_id for movie_id, i in movie_indices.items()}
    
    # Create a mapping of movie IDs to titles
    movie_id_to_title = pd.Series(movies_df['title'].values, index=movies_df['movieId']).to_dict()
    
    return movie_similarity, movie_indices, indices_to_movieid, movie_id_to_title

# Build the collaborative filtering recommender
try:
    movie_similarity, movie_indices, indices_to_movieid, movie_id_to_title = build_collaborative_filtering_recommender(ratings, movies)
    
    def get_collaborative_recommendations(movie_title, movies_df=movies, num_recommendations=10):
        # Find the movieId for the given title
        movie_row = movies_df[movies_df['clean_title'] == movie_title]
        
        if movie_row.empty:
            print(f"'{movie_title}' not found in the dataset. Here are some available titles:")
            sample_titles = movies_df['clean_title'].sample(5).tolist()
            for t in sample_titles:
                print(f"- {t}")
            return None
        
        movie_id = movie_row.iloc[0]['movieId']
        
        # Check if the movie is in the similarity matrix
        if movie_id not in movie_indices:
            print(f"Movie '{movie_title}' has no ratings in the dataset.")
            return None
        
        # Get the index of the movie in the similarity matrix
        idx = movie_indices[movie_id]
        
        # Get the similarity scores
        sim_scores = list(enumerate(movie_similarity[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the most similar movies
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get the movie indices and IDs
        similar_movie_indices = [i[0] for i in sim_scores]
        similar_movie_ids = [indices_to_movieid[idx] for idx in similar_movie_indices]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Get the movie titles
        similar_movie_titles = [movie_id_to_title[movie_id] for movie_id in similar_movie_ids]
        
        # Return the recommendations as a DataFrame
        recommendations = pd.DataFrame({
            'movieId': similar_movie_ids,
            'title': similar_movie_titles,
            'similarity_score': similarity_scores
        })
        
        return recommendations
    
    collaborative_filtering_available = True
except Exception as e:
    print(f"Couldn't build collaborative filtering recommender: {e}")
    collaborative_filtering_available = False

# Function to visualize recommendations
def visualize_recommendations(recommendations, title="Movie Recommendations", recommendation_type="Content-Based"):
    if recommendations is None or len(recommendations) == 0:
        print("No recommendations to visualize.")
        return
    
    # Sort recommendations by similarity score
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.barh(recommendations['title'], recommendations['similarity_score'], color='skyblue')
    
    # Add similarity scores to the end of each bar
    for i, bar in enumerate(bars):
        score = recommendations['similarity_score'].iloc[i]
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
                 va='center', fontsize=10)
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Movie Title')
    plt.title(f'{recommendation_type} Recommendations for: {title}')
    plt.xlim(0, 1.1)  # Set x-axis limit to 0-1 for similarity scores
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Function to get recommendations using both methods
def get_recommendations(movie_title, num_recommendations=10):
    print(f"\nGetting content-based recommendations for '{movie_title}':")
    content_recommendations = get_content_based_recommendations(
        movie_title, num_recommendations=num_recommendations
    )

    if content_recommendations is not None:
        print("\nContent-based recommendations:")
        print(content_recommendations)
        visualize_recommendations(content_recommendations, movie_title, "Content-Based")

    # Declare before usage to avoid UnboundLocalError
    collab_recommendations = None

    if collaborative_filtering_available:
        print(f"\nGetting collaborative filtering recommendations for '{movie_title}':")
        collab_recommendations = get_collaborative_recommendations(
            movie_title, num_recommendations=num_recommendations
        )

        if collab_recommendations is not None:
            print("\nCollaborative filtering recommendations:")
            print(collab_recommendations)
            visualize_recommendations(collab_recommendations, movie_title, "Collaborative Filtering")

# Example usage
movie_title = "Toy Story"  # Example movie title
get_recommendations(movie_title, num_recommendations=10)

# Function to get recommendations for a user
def get_user_recommendations(user_id, num_recommendations=10):
    if not collaborative_filtering_available:
        print("Collaborative filtering is not available.")
        return None
    
    # Get the user's ratings
    user_ratings = ratings[ratings['userId'] == user_id]
    
    if user_ratings.empty:
        print(f"User {user_id} not found in the dataset.")
        return None
    
    # Get the movies the user has rated
    rated_movies = user_ratings['movieId'].tolist()
    
    # Calculate the average rating for each movie
    movie_avg_ratings = ratings.groupby('movieId')['rating'].mean()
    
    # Get the user's average rating
    user_avg_rating = user_ratings['rating'].mean()
    
    # Dict to store potential recommendations
    potential_recommendations = defaultdict(float)
    
    # For each movie the user has rated
    for idx, row in user_ratings.iterrows():
        movie_id = row['movieId']
        user_rating = row['rating']
        
        # If the movie is not in the similarity matrix, skip it
        if movie_id not in movie_indices:
            continue
        
        # Get the index of the movie in the similarity matrix
        movie_idx = movie_indices[movie_id]
        
        # Get similar movies
        similar_movies = [(i, movie_similarity[movie_idx][i]) for i in range(len(movie_similarity[movie_idx]))]
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:21]  # Get top 20 similar movies
        
        # For each similar movie
        for sim_movie_idx, similarity in similar_movies:
            sim_movie_id = indices_to_movieid[sim_movie_idx]
            
            # Skip if the user has already rated this movie
            if sim_movie_id in rated_movies:
                continue
            
            # Calculate the predicted rating
            rating_diff = user_rating - movie_avg_ratings.get(movie_id, user_avg_rating)
            predicted_rating = movie_avg_ratings.get(sim_movie_id, user_avg_rating) + rating_diff * similarity
            
            # Update the potential recommendation score
            potential_recommendations[sim_movie_id] += predicted_rating * similarity
    
    # Sort the recommendations by score
    sorted_recommendations = sorted(potential_recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    if not sorted_recommendations:
        print(f"No recommendations found for user {user_id}.")
        return None
    
    # Create a DataFrame with the recommendations
    recommendation_data = {
        'movieId': [movie_id for movie_id, _ in sorted_recommendations],
        'title': [movie_id_to_title.get(movie_id, f"Unknown ({movie_id})") for movie_id, _ in sorted_recommendations],
        'predicted_rating': [score for _, score in sorted_recommendations]
    }
    
    recommendations = pd.DataFrame(recommendation_data)
    
    return recommendations

# Example usage for user recommendations
if collaborative_filtering_available:
    user_id = 1  # Example user ID
    print(f"\nGetting recommendations for user {user_id}:")
    user_recommendations = get_user_recommendations(user_id, num_recommendations=10)
    
    if user_recommendations is not None:
        print("\nUser recommendations:")
        print(user_recommendations)
        
        # Visualize user recommendations
        plt.figure(figsize=(12, 8))
        bars = plt.barh(user_recommendations['title'], user_recommendations['predicted_rating'], color='lightgreen')
        
        # Add predicted ratings to the end of each bar
        for i, bar in enumerate(bars):
            score = user_recommendations['predicted_rating'].iloc[i]
            plt.text(score + 0.1, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
                     va='center', fontsize=10)
        
        plt.xlabel('Predicted Rating')
        plt.ylabel('Movie Title')
        plt.title(f'Movie Recommendations for User {user_id}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Interactive function to get recommendations
def get_recommendations_for_movie():
    print("\nEnter a movie title to get recommendations:")
    movie_title = input()
    num_recommendations = 10
    
    get_recommendations(movie_title, num_recommendations)

# Interactive function to get user recommendations
def get_recommendations_for_user():
    if not collaborative_filtering_available:
        print("Collaborative filtering is not available.")
        return
    
    print("\nEnter a user ID to get recommendations:")
    try:
        user_id = int(input())
        num_recommendations = 10
        
        user_recommendations = get_user_recommendations(user_id, num_recommendations)
        
        if user_recommendations is not None:
            print("\nUser recommendations:")
            print(user_recommendations)
            
            # Visualize user recommendations
            plt.figure(figsize=(12, 8))
            bars = plt.barh(user_recommendations['title'], user_recommendations['predicted_rating'], color='lightgreen')
            
            # Add predicted ratings to the end of each bar
            for i, bar in enumerate(bars):
                score = user_recommendations['predicted_rating'].iloc[i]
                plt.text(score + 0.1, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
                         va='center', fontsize=10)
            
            plt.xlabel('Predicted Rating')
            plt.ylabel('Movie Title')
            plt.title(f'Movie Recommendations for User {user_id}')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
    except ValueError:
        print("Invalid user ID. Please enter a number.")

print("\nThe recommendation system is ready!")
print("You can use the following functions:")
print("1. get_recommendations(movie_title, num_recommendations=10) - Get recommendations for a movie")
print("2. get_recommendations_for_movie() - Interactive function to get recommendations for a movie")
if collaborative_filtering_available:
    print("3. get_user_recommendations(user_id, num_recommendations=10) - Get recommendations for a user")
    print("4. get_recommendations_for_user() - Interactive function to get recommendations for a user")