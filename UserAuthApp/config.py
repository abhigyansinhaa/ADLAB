"""Database configuration for User Authentication App."""

# MySQL Configuration - Update these values for your MySQL setup
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''  # Set your MySQL password here
MYSQL_DB = 'user_auth_db'
MYSQL_PORT = 3306

# Flask secret key for session management
SECRET_KEY = 'your-secret-key-change-in-production'
