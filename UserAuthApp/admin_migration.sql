-- Admin Panel Migration
-- Run this in MySQL Workbench to add admin support to existing database

USE user_auth_db;

-- Add is_admin column to users table
ALTER TABLE users ADD COLUMN is_admin TINYINT(1) DEFAULT 0;

-- Make first user (or a specific user) an admin - run ONE of these:
-- Option 1: Make user with id=1 an admin
UPDATE users SET is_admin = 1 WHERE id = 1 LIMIT 1;

-- Option 2: Make a specific username admin (uncomment and replace 'admin' with your username)
-- UPDATE users SET is_admin = 1 WHERE username = 'admin';
