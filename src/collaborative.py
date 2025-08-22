# collaborative.py
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from models.models import DatabaseManager
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def collaborative_recommendation(courses: pd.DataFrame, user_ratings: dict, top_n: int = 10, use_ncf: bool = False):
    """
    Enhanced collaborative filtering with matrix factorization (SVD) or Neural Collaborative Filtering
    """
    # Get all ratings from database
    db = DatabaseManager()
    all_ratings = db.get_all_ratings()

    if all_ratings.empty or len(all_ratings) < 10:
        return collaborative_fallback(courses, user_ratings, top_n)

    if use_ncf:
        return ncf_recommendation(courses, all_ratings, user_ratings, top_n)
    else:
        return svd_recommendation(courses, all_ratings, user_ratings, top_n)


def svd_recommendation(courses: pd.DataFrame, all_ratings: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """SVD-based collaborative filtering"""
    # Create user-item matrix
    user_item_matrix = all_ratings.pivot_table(
        index='username', columns='course_id', values='value'
    ).fillna(0)

    # Convert to numpy array
    ratings_matrix = user_item_matrix.values

    # Normalize by each user's mean
    user_ratings_mean = np.mean(ratings_matrix, axis=1)
    ratings_normalized = ratings_matrix - user_ratings_mean.reshape(-1, 1)

    # Perform SVD
    U, sigma, Vt = svds(ratings_normalized, k=min(50, len(user_item_matrix.columns) - 1))
    sigma = np.diag(sigma)

    # Make predictions
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings,
                            columns=user_item_matrix.columns,
                            index=user_item_matrix.index)

    # Get recommendations for current user
    if user_ratings:
        user_vector = pd.DataFrame(0, index=['current_user'], columns=user_item_matrix.columns)
        for course_id, rating in user_ratings.items():
            if course_id in user_vector.columns:
                user_vector.loc['current_user', course_id] = rating

        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(user_vector, user_item_matrix)
        similar_users = np.argsort(similarity[0])[-5:]

        user_preds = preds_df.iloc[similar_users].mean(axis=0)
    else:
        user_preds = preds_df.mean(axis=0)

    user_preds = user_preds[~user_preds.index.isin(user_ratings.keys())]
    top_course_ids = user_preds.sort_values(ascending=False).head(top_n).index.tolist()

    recommendations = courses[courses['Course_ID'].isin(top_course_ids)].copy()
    recommendations['Predicted_Rating'] = recommendations['Course_ID'].map(user_preds)

    return recommendations.sort_values('Predicted_Rating', ascending=False).head(top_n)


def ncf_recommendation(courses: pd.DataFrame, all_ratings: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """
    Neural Collaborative Filtering implementation
    """
    # Prepare data for NCF
    user_encoder = LabelEncoder()
    course_encoder = LabelEncoder()

    # Encode users and courses
    all_ratings['user_encoded'] = user_encoder.fit_transform(all_ratings['username'])
    all_ratings['course_encoded'] = course_encoder.fit_transform(all_ratings['course_id'])

    # Get number of users and courses
    n_users = len(user_encoder.classes_)
    n_courses = len(course_encoder.classes_)

    # Prepare training data
    X = all_ratings[['user_encoded', 'course_encoded']].values
    y = all_ratings['value'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build NCF model
    model = build_ncf_model(n_users, n_courses)

    # Train model
    model.fit([X_train[:, 0], X_train[:, 1]], y_train,
              batch_size=64,
              epochs=10,
              validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
              verbose=0)

    # Get predictions for current user if available
    if user_ratings:
        # Encode current user (assuming username is in session)
        current_user = st.session_state.username if 'username' in st.session_state else 'current_user'

        try:
            user_idx = user_encoder.transform([current_user])[0]
        except:
            # If user not in training data, use average predictions
            return ncf_fallback(courses, all_ratings, user_ratings, top_n)

        # Predict ratings for all courses for this user
        user_array = np.array([user_idx] * n_courses)
        course_array = np.arange(n_courses)

        predictions = model.predict([user_array, course_array], verbose=0).flatten()

        # Get top recommendations
        course_ids = course_encoder.inverse_transform(course_array)
        pred_df = pd.DataFrame({'course_id': course_ids, 'predicted_rating': predictions})

        # Filter out already rated courses
        pred_df = pred_df[~pred_df['course_id'].isin(user_ratings.keys())]

        top_course_ids = pred_df.nlargest(top_n, 'predicted_rating')['course_id'].tolist()

        recommendations = courses[courses['Course_ID'].isin(top_course_ids)].copy()
        recommendations['Predicted_Rating'] = recommendations['Course_ID'].map(
            dict(zip(pred_df['course_id'], pred_df['predicted_rating']))
        )

        return recommendations.sort_values('Predicted_Rating', ascending=False).head(top_n)

    else:
        # For new users without ratings, use popularity-based fallback
        return ncf_fallback(courses, all_ratings, user_ratings, top_n)


def build_ncf_model(n_users, n_courses, embedding_size=50):
    """
    Build Neural Collaborative Filtering model
    """
    # User embedding
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten()(user_embedding)

    # Course embedding
    course_input = Input(shape=(1,), name='course_input')
    course_embedding = Embedding(n_courses, embedding_size, name='course_embedding')(course_input)
    course_vec = Flatten()(course_embedding)

    # Concatenate embeddings
    concat = Concatenate()([user_vec, course_vec])

    # Add dense layers
    fc1 = Dense(128, activation='relu')(concat)
    dropout1 = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(fc2)
    fc3 = Dense(32, activation='relu')(dropout2)

    # Output layer
    output = Dense(1, activation='sigmoid')(fc3)

    # Compile model
    model = Model(inputs=[user_input, course_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def ncf_fallback(courses: pd.DataFrame, all_ratings: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """Fallback for NCF when user is new or not in training data"""
    # Use average ratings as fallback
    avg_ratings = all_ratings.groupby('course_id')['value'].mean().reset_index()
    avg_ratings = avg_ratings[~avg_ratings['course_id'].isin(user_ratings.keys())]

    top_course_ids = avg_ratings.nlargest(top_n, 'value')['course_id'].tolist()

    recommendations = courses[courses['Course_ID'].isin(top_course_ids)].copy()
    recommendations['Predicted_Rating'] = recommendations['Course_ID'].map(
        dict(zip(avg_ratings['course_id'], avg_ratings['value']))
    )

    return recommendations.sort_values('Predicted_Rating', ascending=False).head(top_n)


def collaborative_fallback(courses: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """Fallback collaborative filtering"""
    rated_titles = set(user_ratings.keys())
    ranked = courses.sort_values(by=["Rate", "Reviews"], ascending=[False, False])
    ranked = ranked[~ranked["Course_ID"].isin(rated_titles)]
    return ranked.head(top_n)