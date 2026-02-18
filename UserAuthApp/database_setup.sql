-- User Authentication & Document Sharing Database Setup
-- Run this script in MySQL Workbench to create the database and tables

CREATE DATABASE IF NOT EXISTS user_auth_db;
USE user_auth_db;

-- Users table: stores authentication and personal details
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Grades table: stores user grades (admin/managed, users cannot edit)
CREATE TABLE IF NOT EXISTS grades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    subject VARCHAR(100) NOT NULL,
    marks DECIMAL(5,2) NOT NULL,
    max_marks DECIMAL(5,2) DEFAULT 100,
    grade VARCHAR(10),
    semester VARCHAR(20),
    academic_year VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_grades (user_id)
);

-- Shared documents table (optional for document sharing feature)
CREATE TABLE IF NOT EXISTS shared_documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    shared_with TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Insert sample grades for testing (run after creating your first user)
-- Replace 1 with actual user_id after signup
-- INSERT INTO grades (user_id, subject, marks, max_marks, grade, semester, academic_year) VALUES
-- (1, 'Mathematics', 85, 100, 'A', '1', '2024-25'),
-- (1, 'Physics', 78, 100, 'B+', '1', '2024-25'),
-- (1, 'Computer Science', 92, 100, 'A+', '1', '2024-25');
