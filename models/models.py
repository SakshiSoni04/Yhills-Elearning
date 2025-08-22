# models.py
import mysql.connector
import pandas as pd
from typing import List, Dict, Any, Optional
import hashlib
import json
from mysql.connector import Error


class DatabaseManager:
    def __init__(self, host="localhost", database="elearning_db",
                 user="root", password="Sahil@2006", port=3306):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.init_db()

    def get_connection(self):
        """Create and return a database connection"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            return connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None

    def init_db(self):
        """Initialize database tables"""
        connection = self.get_connection()
        if connection is None:
            print("Failed to initialize database")
            return

        cursor = connection.cursor()

        try:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # User interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    course_id VARCHAR(255) NOT NULL,
                    interaction_type ENUM('rating', 'view', 'enroll') NOT NULL,
                    value FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_course (user_id, course_id)
                )
            ''')

            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    preferred_subjects TEXT,
                    preferred_levels TEXT,
                    skill_interests TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE KEY unique_user (user_id)
                )
            ''')

            connection.commit()
            print("Database initialized successfully")

        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            cursor.close()
            connection.close()

    def create_user(self, username: str, password: str) -> int:
        """Create a new user and return user ID"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        connection = self.get_connection()
        if connection is None:
            return -1

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash)
            )
            user_id = cursor.lastrowid
            connection.commit()
            return user_id
        except Error as e:
            print(f"Error creating user: {e}")
            return -1
        finally:
            cursor.close()
            connection.close()

    def verify_user(self, username: str, password: str) -> int:
        """Verify user credentials and return user ID if valid"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        connection = self.get_connection()
        if connection is None:
            return -1

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")
            cursor.execute(
                "SELECT id FROM users WHERE username = %s AND password_hash = %s",
                (username, password_hash)
            )

            result = cursor.fetchone()
            return result[0] if result else -1
        except Error as e:
            print(f"Error verifying user: {e}")
            return -1
        finally:
            cursor.close()
            connection.close()

    def add_interaction(self, user_id: int, course_id: str, interaction_type: str, value: float = None):
        """Record a user interaction with a course"""
        connection = self.get_connection()
        if connection is None:
            return

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")
            cursor.execute(
                "INSERT INTO user_interactions (user_id, course_id, interaction_type, value) VALUES (%s, %s, %s, %s)",
                (user_id, course_id, interaction_type, value)
            )

            connection.commit()
        except Error as e:
            print(f"Error adding interaction: {e}")
        finally:
            cursor.close()
            connection.close()

    def get_user_ratings(self, user_id: int) -> Dict[str, float]:
        """Get all ratings for a user"""
        connection = self.get_connection()
        if connection is None:
            return {}

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")
            cursor.execute(
                "SELECT course_id, value FROM user_interactions WHERE user_id = %s AND interaction_type = 'rating'",
                (user_id,)
            )

            ratings = {row[0]: row[1] for row in cursor.fetchall()}
            return ratings
        except Error as e:
            print(f"Error getting user ratings: {e}")
            return {}
        finally:
            cursor.close()
            connection.close()

    def get_all_ratings(self) -> pd.DataFrame:
        """Get all ratings from all users as a DataFrame"""
        connection = self.get_connection()
        if connection is None:
            return pd.DataFrame()

        try:
            query = """
                SELECT u.username, ui.course_id, ui.value 
                FROM user_interactions ui
                JOIN users u ON ui.user_id = u.id
                WHERE ui.interaction_type = 'rating'
            """

            ratings_df = pd.read_sql_query(query, connection)
            return ratings_df
        except Error as e:
            print(f"Error getting all ratings: {e}")
            return pd.DataFrame()
        finally:
            connection.close()

    def save_user_profile(self, user_id: int, profile_data: Dict[str, Any]):
        """Save or update user profile"""
        connection = self.get_connection()
        if connection is None:
            return

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")

            # Convert lists to JSON strings
            preferred_subjects = json.dumps(profile_data.get('preferred_subjects', []))
            preferred_levels = json.dumps(profile_data.get('preferred_levels', []))
            skill_interests = json.dumps(profile_data.get('skill_interests', []))

            cursor.execute(
                """INSERT INTO user_profiles (user_id, preferred_subjects, preferred_levels, skill_interests) 
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE 
                   preferred_subjects = VALUES(preferred_subjects),
                   preferred_levels = VALUES(preferred_levels),
                   skill_interests = VALUES(skill_interests)""",
                (user_id, preferred_subjects, preferred_levels, skill_interests)
            )

            connection.commit()
        except Error as e:
            print(f"Error saving user profile: {e}")
        finally:
            cursor.close()
            connection.close()

    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        connection = self.get_connection()
        if connection is None:
            return None

        cursor = connection.cursor()

        try:
            cursor.execute(f"USE {self.database}")
            cursor.execute(
                "SELECT preferred_subjects, preferred_levels, skill_interests FROM user_profiles WHERE user_id = %s",
                (user_id,)
            )

            result = cursor.fetchone()
            if result:
                return {
                    'preferred_subjects': json.loads(result[0]) if result[0] else [],
                    'preferred_levels': json.loads(result[1]) if result[1] else [],
                    'skill_interests': json.loads(result[2]) if result[2] else []
                }
            return None
        except Error as e:
            print(f"Error getting user profile: {e}")
            return None
        finally:
            cursor.close()
            connection.close()