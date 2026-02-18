"""
User Authentication & Document Sharing Application
Flask + MySQL integration with signup, login, profile management, and grades display.
"""

import os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load config from config.py if available, else use env/defaults
try:
    from config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT, SECRET_KEY
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['MYSQL_HOST'] = MYSQL_HOST
    app.config['MYSQL_USER'] = MYSQL_USER
    app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
    app.config['MYSQL_DB'] = MYSQL_DB
    app.config['MYSQL_PORT'] = MYSQL_PORT
except ImportError:
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST', 'localhost')
    app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER', 'root')
    app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', '')
    app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB', 'user_auth_db')
    app.config['MYSQL_PORT'] = int(os.environ.get('MYSQL_PORT', 3306))

# Upload folder for document sharing
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'xlsx', 'xls'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mysql = MySQL(app)


def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# Authentication Routes
# ============================================

@app.route('/')
def index():
    """Landing page - redirect to login or dashboard."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for existing users."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.', 'danger')
            return render_template('login.html')

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, password, full_name FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['full_name'] = user[3] or user[1]
            flash(f'Welcome back, {session["full_name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Registration form for new users."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()

        errors = []
        if not username:
            errors.append('Username is required.')
        if not email:
            errors.append('Email is required.')
        if not password:
            errors.append('Password is required.')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('signup.html')

        hashed = generate_password_hash(password, method='scrypt')

        cur = mysql.connection.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, password, full_name) VALUES (%s, %s, %s, %s)",
                (username, email, hashed, full_name or username)
            )
            mysql.connection.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            mysql.connection.rollback()
            if 'Duplicate' in str(e) or 'username' in str(e).lower():
                flash('Username already exists.', 'danger')
            elif 'email' in str(e).lower():
                flash('Email already registered.', 'danger')
            else:
                flash(f'Registration failed: {str(e)}', 'danger')
            return render_template('signup.html')
        finally:
            cur.close()

    return render_template('signup.html')


@app.route('/logout')
def logout():
    """Logout user."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============================================
# Dashboard & Protected Routes
# ============================================

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard after login."""
    return render_template('dashboard.html')


# ============================================
# Profile Management
# ============================================

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View and update personal details."""
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT username, email, full_name, phone, address FROM users WHERE id = %s",
        (session['user_id'],)
    )
    user = cur.fetchone()
    cur.close()

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE users SET full_name = %s, phone = %s, address = %s WHERE id = %s",
            (full_name, phone, address, session['user_id'])
        )
        mysql.connection.commit()
        cur.close()

        session['full_name'] = full_name or session['username']
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user={
        'username': user[0],
        'email': user[1],
        'full_name': user[2] or '',
        'phone': user[3] or '',
        'address': user[4] or '',
    })


@app.route('/reset-password', methods=['GET', 'POST'])
@login_required
def reset_password():
    """Reset password for logged-in user."""
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        cur = mysql.connection.cursor()
        cur.execute("SELECT password FROM users WHERE id = %s", (session['user_id'],))
        row = cur.fetchone()
        cur.close()

        if not row or not check_password_hash(row[0], current_password):
            flash('Current password is incorrect.', 'danger')
            return render_template('reset_password.html')

        if len(new_password) < 6:
            flash('New password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html')

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return render_template('reset_password.html')

        hashed = generate_password_hash(new_password, method='scrypt')
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET password = %s WHERE id = %s", (hashed, session['user_id']))
        mysql.connection.commit()
        cur.close()

        flash('Password updated successfully!', 'success')
        return redirect(url_for('profile'))

    return render_template('reset_password.html')


# ============================================
# Grades (Read-Only)
# ============================================

@app.route('/grades')
@login_required
def grades():
    """Display user grades (read-only)."""
    cur = mysql.connection.cursor()
    cur.execute(
        """SELECT subject, marks, max_marks, grade, semester, academic_year 
           FROM grades WHERE user_id = %s ORDER BY semester, subject""",
        (session['user_id'],)
    )
    grades_list = cur.fetchall()
    cur.close()

    grades_data = [
        {
            'subject': row[0],
            'marks': float(row[1]),
            'max_marks': float(row[2]),
            'grade': row[3] or '-',
            'semester': row[4] or '-',
            'academic_year': row[5] or '-',
        }
        for row in grades_list
    ]

    return render_template('grades.html', grades=grades_data)


# ============================================
# Document Sharing
# ============================================

@app.route('/documents', methods=['GET', 'POST'])
@login_required
def documents():
    """Upload and view shared documents."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(url_for('documents'))

        file = request.files['file']
        title = request.form.get('title', '').strip() or file.filename

        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('documents'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO shared_documents (user_id, title, filename, file_path) VALUES (%s, %s, %s, %s)",
                (session['user_id'], title, filename, filepath)
            )
            mysql.connection.commit()
            cur.close()

            flash(f'Document "{title}" uploaded successfully!', 'success')
        else:
            flash('Invalid file type. Allowed: pdf, doc, docx, txt, xlsx, xls', 'danger')

        return redirect(url_for('documents'))

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT id, title, filename, created_at FROM shared_documents WHERE user_id = %s ORDER BY created_at DESC",
        (session['user_id'],)
    )
    docs = cur.fetchall()
    cur.close()

    documents_list = [
        {'id': d[0], 'title': d[1], 'filename': d[2], 'created_at': str(d[3])}
        for d in docs
    ]

    return render_template('documents.html', documents=documents_list)


# ============================================
# Admin: Add sample grades (for testing)
# ============================================

@app.route('/admin/add-sample-grades')
@login_required
def add_sample_grades():
    """Add sample grades for testing - only if user has none."""
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM grades WHERE user_id = %s", (session['user_id'],))
    count = cur.fetchone()[0]
    if count > 0:
        flash('You already have grades. Sample grades not added.', 'info')
        cur.close()
        return redirect(url_for('grades'))

    sample = [
        (session['user_id'], 'Mathematics', 85, 100, 'A', '1', '2024-25'),
        (session['user_id'], 'Physics', 78, 100, 'B+', '1', '2024-25'),
        (session['user_id'], 'Computer Science', 92, 100, 'A+', '1', '2024-25'),
        (session['user_id'], 'English', 88, 100, 'A', '1', '2024-25'),
    ]
    cur.executemany(
        "INSERT INTO grades (user_id, subject, marks, max_marks, grade, semester, academic_year) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        sample
    )
    mysql.connection.commit()
    cur.close()
    flash('Sample grades added. View them in the Grades section.', 'success')
    return redirect(url_for('grades'))


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("User Auth & Document Sharing - http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
