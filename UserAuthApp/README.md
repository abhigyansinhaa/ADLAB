# User Authentication & Document Sharing App

A Flask application with MySQL for user authentication, profile management, grades display, and document sharing.

## Setup Instructions

### 1. Install MySQL Workbench

- Download and install [MySQL Workbench](https://dev.mysql.com/downloads/workbench/) for your system.
- Ensure MySQL Server is running.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** `flask-mysqldb` requires MySQL client libraries. On Windows, install [MySQL Connector](https://dev.mysql.com/downloads/connector/python/) or use a pre-built wheel. If installation fails, try:
`pip install mysqlclient` (may need Visual C++ Build Tools on Windows).

### 3. Create the Database

1. Open MySQL Workbench and connect to your MySQL server.
2. Open the `database_setup.sql` file.
3. Execute the script to create the `user_auth_db` database and tables.

Or run from command line:

```bash
mysql -u root -p < database_setup.sql
```

### 4. Configure Database Connection

Edit `app.py` or set environment variables:

- `MYSQL_HOST` (default: localhost)
- `MYSQL_USER` (default: root)
- `MYSQL_PASSWORD` (your MySQL password)
- `MYSQL_DB` (default: user_auth_db)

### 5. Run the Application

```bash
python app.py
```

Visit **http://localhost:5000**

## Features

- **Signup**: New users create accounts with username, email, and password
- **Login**: Existing users log in with credentials
- **Profile**: Update full name, phone, and address
- **Reset Password**: Change password (requires current password)
- **Grades**: View marks (read-only; users cannot edit)
- **Documents**: Upload and manage shared documents
- **Responsive Frontend**: Works on desktop and mobile devices

## Adding Grades (Admin)

Grades are stored in the `grades` table. To add grades for a user:

1. Find the user's `id` from the `users` table.
2. Insert records into `grades`:

```sql
INSERT INTO grades (user_id, subject, marks, max_marks, grade, semester, academic_year)
VALUES (1, 'Mathematics', 85, 100, 'A', '1', '2024-25');
```

Or use the "Add Sample Grades (Demo)" button on the dashboard after login for testing.
