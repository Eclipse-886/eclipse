from flask import Flask, request, render_template_string, redirect, url_for, session
import requests
import json
import os
import re
import sqlite3
from datetime import datetime
import io
from pdfminer.high_level import extract_text as extract_text_pdfminer
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from datasets import load_dataset
from cryptography.fernet import Fernet

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# ---------- Encryption ----------
KEY_FILE = "secret.key"
def get_cipher():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        print("New encryption key generated.")
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return Fernet(key)

cipher = get_cipher()

def encrypt_text(text: str) -> str:
    if not text:
        return ""
    return cipher.encrypt(text.encode()).decode()

def decrypt_text(encrypted: str) -> str:
    if not encrypted:
        return ""
    return cipher.decrypt(encrypted.encode()).decode()

# ---------- Database ----------
DB_PATH = "resume_analyzer.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        company_id INTEGER NOT NULL,
        FOREIGN KEY (company_id) REFERENCES companies (id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        job_descs TEXT NOT NULL,
        department TEXT NOT NULL,
        FOREIGN KEY (company_id) REFERENCES companies (id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        company_id INTEGER NOT NULL,
        candidate_id TEXT NOT NULL,
        department TEXT NOT NULL,
        filename TEXT NOT NULL,
        extracted_text TEXT,
        best_score INTEGER,
        best_job TEXT,
        explanation TEXT,
        text_preview TEXT,
        score_details TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
        FOREIGN KEY (company_id) REFERENCES companies (id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS resume_library (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        candidate_id TEXT NOT NULL,
        department TEXT NOT NULL,
        filename TEXT NOT NULL,
        extracted_text TEXT NOT NULL,
        text_preview TEXT NOT NULL,
        upload_time TEXT NOT NULL,
        UNIQUE(company_id, department, candidate_id),
        FOREIGN KEY (company_id) REFERENCES companies (id)
    )''')
    # Insert sample companies and users (passwords plain text for demo)
    c.execute("INSERT OR IGNORE INTO companies (name) VALUES ('TechCorp')")
    c.execute("INSERT OR IGNORE INTO companies (name) VALUES ('FinanceInc')")
    c.execute("INSERT OR IGNORE INTO users (username, password, company_id) VALUES ('hr_tech', 'tech123', (SELECT id FROM companies WHERE name='TechCorp'))")
    c.execute("INSERT OR IGNORE INTO users (username, password, company_id) VALUES ('hr_finance', 'fin123', (SELECT id FROM companies WHERE name='FinanceInc'))")
    conn.commit()
    conn.close()

init_db()

def get_current_company_id():
    return session.get('company_id')

def is_logged_in():
    return get_current_company_id() is not None

# ---------- Config ----------
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # adjust if needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Scoring weights per department ----------
INDUSTRY_WEIGHTS = {
    "technology": {"name": "Technology", "weights": {"skills": 0.40, "experience": 0.30, "education": 0.20, "certificate": 0.10}},
    "marketing":   {"name": "Marketing",   "weights": {"skills": 0.30, "experience": 0.35, "education": 0.15, "certificate": 0.20}},
    "finance":     {"name": "Finance",     "weights": {"skills": 0.25, "experience": 0.40, "education": 0.20, "certificate": 0.15}},
    "sales":       {"name": "Sales",       "weights": {"skills": 0.35, "experience": 0.40, "education": 0.10, "certificate": 0.15}}
}

# ---------- Ollama helpers ----------
def match_with_ollama_detailed(resume_text, job_desc, industry_key):
    weights = INDUSTRY_WEIGHTS.get(industry_key, INDUSTRY_WEIGHTS["technology"])["weights"]
    weight_text = f"skills {weights['skills']*100}%, experience {weights['experience']*100}%, education {weights['education']*100}%, certificate {weights['certificate']*100}%."
    prompt = f"""Please assess the match between the following resume and the job description, and calculate scores according to the given weights.
Resume content:
{resume_text[:5000]}

Job description:
{job_desc}

Scoring weights: {weight_text}
Evaluate four aspects: skills, experience, education, certificate. Give each a score from 0 to 100, then compute the weighted total score.
Output format must be strictly JSON:
{{"skills_score": score, "experience_score": score, "education_score": score, "certificate_score": score, "total_score": total}}
Output only JSON, no extra text.
"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": "gemma3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json()['response']
        cleaned = raw.replace('*', '').strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            details = json.loads(json_match.group())
            total = details.get("total_score", 0)
            for key in ["skills_score", "experience_score", "education_score", "certificate_score"]:
                if key not in details:
                    details[key] = 0
            return total, details
        else:
            return 0, {"skills_score": 0, "experience_score": 0, "education_score": 0, "certificate_score": 0, "total_score": 0}
    except Exception as e:
        print("Ollama detailed error:", e)
        return 0, {"skills_score": 0, "experience_score": 0, "education_score": 0, "certificate_score": 0, "total_score": 0}

def match_with_ollama_simple(resume_text, job_desc):
    prompt = f"""Please assess the degree of match between the following resume and the job description.
Resume Content:
{resume_text[:5000]}

Job Description:
{job_desc}

Provide a matching percentage (0-100) and a brief explanation.
Output exactly two lines:
Score: XX
Explanation: XXXXX
"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": "gemma3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json()['response']
        cleaned = raw.replace('*', '')
        score = None
        explanation = cleaned
        for line in cleaned.split('\n'):
            if 'Score' in line:
                nums = re.findall(r'\d+', line)
                if nums:
                    score = int(nums[0])
            elif 'Explanation' in line:
                explanation = line.split('Explanation')[-1].replace('：', '').replace(':', '').strip()
        return (score if score is not None else 0), explanation
    except Exception as e:
        print("Ollama simple error:", e)
        return 0, "Error"

# ---------- Text extraction (no file saved) ----------
def clean_messy_text(text):
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\!\?\;\:\-\+\/\=\[\]\(\)\%\#\@\&\*]', '', text)
    return text

def extract_text_from_pdf_bytes(file_bytes):
    try:
        text = extract_text_pdfminer(io.BytesIO(file_bytes))
        if text and len(text.strip()) > 100:
            return clean_messy_text(text)
    except: pass
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
        if text and len(text.strip()) > 100:
            return clean_messy_text(text)
    except: pass
    try:
        images = convert_from_bytes(file_bytes, dpi=200, first_page=1, last_page=5)
        text = '\n'.join([pytesseract.image_to_string(img, lang='eng+chi_sim') for img in images])
        if text and len(text.strip()) > 50:
            return clean_messy_text(text)
    except: pass
    return ""

def extract_text_from_image_bytes(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang='eng+chi_sim')
        return clean_messy_text(text)
    except:
        return ""

def get_text_from_file_bytes(filename, file_bytes):
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf_bytes(file_bytes)
    elif ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image_bytes(file_bytes)
    return ""

# ---------- Library operations with company isolation ----------
def add_to_library(company_id, candidate_id, filename, department, extracted_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    encrypted_text = encrypt_text(extracted_text)
    encrypted_preview = encrypt_text(extracted_text[:400] + ('...' if len(extracted_text) > 400 else ''))
    c.execute('''INSERT OR REPLACE INTO resume_library 
                 (company_id, candidate_id, department, filename, extracted_text, text_preview, upload_time)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (company_id, candidate_id, department, filename, encrypted_text, encrypted_preview, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_resumes_by_company(company_id, department=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if department:
        c.execute("SELECT candidate_id, filename, extracted_text, text_preview, upload_time FROM resume_library WHERE company_id=? AND department=?", (company_id, department))
    else:
        c.execute("SELECT candidate_id, filename, extracted_text, text_preview, upload_time FROM resume_library WHERE company_id=?", (company_id,))
    rows = c.fetchall()
    conn.close()
    resumes = []
    for candidate_id, filename, enc_text, enc_preview, upload_time in rows:
        resumes.append((candidate_id, filename, decrypt_text(enc_text), decrypt_text(enc_preview), upload_time))
    return resumes

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Login</title><style>body{font-family:sans-serif;background:#f0f2f5;display:flex;justify-content:center;align-items:center;height:100vh;}.card{background:white;border-radius:28px;padding:40px;width:350px;text-align:center;}input{width:100%;padding:10px;margin:10px 0;border-radius:16px;border:1px solid #ccc;}button{background:#2563eb;color:white;border:none;padding:10px;width:100%;border-radius:40px;cursor:pointer;}.error{color:red;}</style></head>
<body><div class="card"><h2>AI Resume Analyzer</h2><form method="post"><input type="text" name="username" placeholder="Username" required><input type="password" name="password" placeholder="Password" required><button type="submit">Login</button>{% if error %}<div class="error">{{ error }}</div>{% endif %}</form><p>Don't have an account? <a href="/register">Register here</a></p><p style="font-size:0.8rem;">Demo: hr_tech / tech123 &nbsp;|&nbsp; hr_finance / fin123</p></div></body></html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Register</title><style>body{font-family:sans-serif;background:#f0f2f5;display:flex;justify-content:center;align-items:center;height:100vh;}.card{background:white;border-radius:28px;padding:40px;width:400px;text-align:center;}input,select{width:100%;padding:10px;margin:10px 0;border-radius:16px;border:1px solid #ccc;}button{background:#2563eb;color:white;border:none;padding:10px;width:100%;border-radius:40px;cursor:pointer;}.error{color:red;}</style></head>
<body>
<div class="card">
<h2>Register HR Account</h2>
<form method="post">
<input type="text" name="username" placeholder="Username" required>
<input type="password" name="password" placeholder="Password" required>
<select name="company_id" required>
    <option value="">Select your company</option>
    {% for company in companies %}
    <option value="{{ company.id }}">{{ company.name }}</option>
    {% endfor %}
</select>
<button type="submit">Register</button>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
</form>
<p><a href="/login">Back to Login</a></p>
</div>
</body>
</html>
'''

MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>AI Resume Analyzer</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box;}body{font-family:'Inter',sans-serif;background:linear-gradient(135deg,#f5f7fa 0%,#e9edf2 100%);padding:40px 24px;color:#1e293b;}.container{max-width:1400px;margin:0 auto;}.card{background:white;border-radius:28px;box-shadow:0 20px 35px -10px rgba(0,0,0,0.08);overflow:hidden;margin-bottom:36px;}.card-header{padding:24px 32px 0 32px;font-weight:700;font-size:1.5rem;border-bottom:2px solid #eef2ff;display:flex;align-items:center;gap:12px;}.card-header i{color:#3b82f6;font-size:1.8rem;}.card-body{padding:24px 32px 32px 32px;}.nav-bar{background:white;border-radius:60px;padding:12px 28px;display:flex;justify-content:space-between;margin-bottom:32px;}.logo{font-weight:800;font-size:1.5rem;background:linear-gradient(135deg,#2563eb,#7c3aed);-webkit-background-clip:text;background-clip:text;color:transparent;}.nav-links a{text-decoration:none;margin-left:28px;color:#4b5563;font-weight:500;}input,textarea,select{width:100%;padding:12px 16px;border:1px solid #d1d5db;border-radius:16px;margin-bottom:16px;}button{background:#2563eb;color:white;border:none;padding:12px 28px;border-radius:40px;cursor:pointer;}.table-responsive{overflow-x:auto;}table{width:100%;border-collapse:collapse;}th,td{padding:12px;border-bottom:1px solid #e2e8f0;text-align:left;}.rank-badge{display:inline-block;width:32px;height:32px;line-height:32px;text-align:center;border-radius:30px;font-weight:700;}.rank-1{background:#ffd966;}.rank-2{background:#e2e8f0;}.rank-3{background:#fce5b4;}.score-high{color:#10b981;font-weight:bold;}.detail-btn{background:#8b5cf6;color:white;border:none;padding:4px 12px;border-radius:20px;cursor:pointer;}.score-detail{background:#f8fafc;padding:10px;border-radius:12px;margin-top:8px;display:none;}.footer-note{text-align:center;margin-top:32px;color:#6b7280;}
</style>
<script>function toggleDetail(id){var div=document.getElementById('detail-'+id);div.style.display=div.style.display==='none'?'block':'none';}</script>
</head>
<body>
<div class="container">
<div class="nav-bar"><div class="logo">AI Resume Analyzer</div><div class="nav-links"><a href="/">Home</a><a href="/history">History</a><a href="/library">Library</a><a href="/match_from_library">Match</a><a href="/logout">Logout</a></div></div>
<div class="card"><div class="card-body"><strong>Company:</strong> {{ company_name }}</div></div>
<div class="card"><div class="card-header">Upload & Analyze</div><div class="card-body">
<form method="post" action="/upload" enctype="multipart/form-data">
<label>Candidate ID (email or name):</label><input type="text" name="candidate_id" placeholder="e.g., john@example.com" required>
<label>Select department:</label><select name="department"><option value="technology">Technology</option><option value="marketing">Marketing</option><option value="finance">Finance</option><option value="sales">Sales</option></select>
<label>Upload resume (PDF/PNG/JPG):</label><input type="file" name="file" accept=".pdf,.png,.jpg,.jpeg" required>
<label>Job descriptions (one per line):</label><textarea name="job_descs" rows="5" placeholder="e.g.&#10;Python developer with Flask&#10;Data scientist with SQL"></textarea>
<button type="submit">Analyze & Rank</button>
</form></div></div>
<div class="card"><div class="card-header">Test with HuggingFace Dataset</div><div class="card-body"><form method="post" action="/test_huggingface_dataset"><label>Resumes (max 10): <input type="number" name="num_resumes" value="5" min="1" max="10"></label><label>Jobs (max 5): <input type="number" name="num_jobs" value="3" min="1" max="5"></label><button type="submit">Load & Test</button></form></div></div>
{% if candidates %}
<div class="card"><div class="card-header">Top {{ candidates|length }} Candidates</div><div class="card-body"><div class="table-responsive"><table><thead><tr><th>Rank</th><th>Candidate ID</th><th>Filename</th><th>Best Job</th><th>Score</th><th>Details</th><th>Preview</th></tr></thead><tbody>{% for c in candidates %}<tr><td><span class="rank-badge rank-{{ loop.index if loop.index<=3 else 'other' }}">{{ loop.index }}</span></td><td>{{ c.candidate_id }}</td><td>{{ c.filename }}</td><td>{{ c.best_job }}</td><td class="score-high">{{ c.score }}%</td><td><button class="detail-btn" onclick="toggleDetail('{{ loop.index }}')">View Details</button><div id="detail-{{ loop.index }}" class="score-detail">Skills: {{ c.details.skills_score }}<br>Experience: {{ c.details.experience_score }}<br>Education: {{ c.details.education_score }}<br>Certificate: {{ c.details.certificate_score }}</div></td><td><pre>{{ c.text_preview[:150] }}...</pre></td></tr>{% endfor %}</tbody></table></div></div></div>
{% endif %}
{% if error %}<div class="card"><div class="card-body" style="color:red;">Error: {{ error }}</div></div>{% endif %}
<div class="footer-note">Files not saved · Encrypted · Multi‑tenant (HR sees only own company) · Same (company,dept,candidate) overwrites old CV</div>
</div>
</body>
</html>
'''

HISTORY_TEMPLATE = '''
<!DOCTYPE html><html><head><title>History</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><style>body{font-family:'Inter',sans-serif;background:#f0f2f5;padding:40px;}.container{max-width:1200px;margin:0 auto;}.nav-bar{background:white;border-radius:60px;padding:12px 28px;display:flex;justify-content:space-between;margin-bottom:32px;}.card{background:white;border-radius:28px;padding:24px;}table{width:100%;border-collapse:collapse;}th,td{padding:12px;border-bottom:1px solid #e2e8f0;}.btn{background:#eef2ff;padding:6px 14px;border-radius:40px;text-decoration:none;margin-right:8px;}</style></head><body><div class="container"><div class="nav-bar"><div class="logo">AI Resume Analyzer</div><div class="nav-links"><a href="/">Home</a><a href="/history">History</a><a href="/library">Library</a><a href="/match_from_library">Match</a><a href="/logout">Logout</a></div></div><div class="card"><h2>Upload Sessions</h2><div style="overflow-x:auto;"><table><thead><tr><th>ID</th><th>Time</th><th>Department</th><th>Job Descriptions</th><th>Actions</th></tr></thead><tbody>{% for s in sessions %}<tr><td>{{ s.id }}</td><td>{{ s.timestamp[:19] }}</td><td>{{ s.department }}</td><td>{{ s.job_descs[:80] }}...</td><td><a href="/view_session/{{ s.id }}" class="btn">View</a><a href="/delete_session/{{ s.id }}" class="btn" style="background:#fee2e2;color:#b91c1c;" onclick="return confirm('Delete?')">Delete</a></td></tr>{% endfor %}</tbody></table></div></div></div></body></html>
'''

LIBRARY_TEMPLATE = '''
<!DOCTYPE html><html><head><title>Library</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><style>body{font-family:'Inter',sans-serif;background:#f0f2f5;padding:40px;}.container{max-width:1200px;margin:0 auto;}.nav-bar{background:white;border-radius:60px;padding:12px 28px;display:flex;justify-content:space-between;margin-bottom:32px;}.card{background:white;border-radius:28px;padding:24px;}table{width:100%;border-collapse:collapse;}th,td{padding:12px;border-bottom:1px solid #e2e8f0;}.btn-danger{background:#fee2e2;padding:6px 14px;border-radius:40px;text-decoration:none;color:#b91c1c;}select{padding:8px;border-radius:20px;margin-bottom:20px;}</style></head><body><div class="container"><div class="nav-bar"><div class="logo">AI Resume Analyzer</div><div class="nav-links"><a href="/">Home</a><a href="/history">History</a><a href="/library">Library</a><a href="/match_from_library">Match</a><a href="/logout">Logout</a></div></div><div class="card"><h2>Resume Library</h2><form method="get" action="/library"><label>Filter by department:</label> <select name="dept_filter"><option value="all">All</option><option value="technology">Technology</option><option value="marketing">Marketing</option><option value="finance">Finance</option><option value="sales">Sales</option></select> <button type="submit">Filter</button></form><div style="overflow-x:auto;"><table><thead><tr><th>Candidate ID</th><th>Filename</th><th>Department</th><th>Upload Time</th><th>Preview</th><th>Action</th></tr></thead><tbody>{% for r in resumes %}<tr><td>{{ r.candidate_id }}</td><td>{{ r.filename }}</td><td>{{ r.department }}</td><td>{{ r.upload_time[:19] }}</td><td><pre>{{ r.text_preview }}</pre></td><td><a href="/delete_from_library/{{ r.id }}" class="btn-danger" onclick="return confirm('Delete?')">Delete</a></td></tr>{% endfor %}</tbody></table></div></div></div></body></html>
'''

MATCH_TEMPLATE = '''
<!DOCTYPE html><html><head><title>Match from Library</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><style>body{font-family:'Inter',sans-serif;background:#f0f2f5;padding:40px;}.container{max-width:1200px;margin:0 auto;}.nav-bar{background:white;border-radius:60px;padding:12px 28px;display:flex;justify-content:space-between;margin-bottom:32px;}.card{background:white;border-radius:28px;padding:24px;margin-bottom:24px;}button{background:#2563eb;color:white;border:none;padding:12px 28px;border-radius:40px;cursor:pointer;}textarea,select{width:100%;border-radius:16px;border:1px solid #ccc;padding:12px;margin-bottom:16px;}table{width:100%;border-collapse:collapse;}th,td{padding:12px;border-bottom:1px solid #e2e8f0;}.score-high{color:#10b981;font-weight:bold;}</style></head><body><div class="container"><div class="nav-bar"><div class="logo">AI Resume Analyzer</div><div class="nav-links"><a href="/">Home</a><a href="/history">History</a><a href="/library">Library</a><a href="/match_from_library">Match</a><a href="/logout">Logout</a></div></div><div class="card"><h2>Match from Library (HR: your department only)</h2><form method="post"><textarea name="job_desc" rows="5" placeholder="Enter job description..." required></textarea><label>Your department (only resumes from this department will be used):</label><select name="dept"><option value="technology">Technology</option><option value="marketing">Marketing</option><option value="finance">Finance</option><option value="sales">Sales</option></select><button type="submit">Find Top 10</button></form></div>{% if candidates %}<div class="card"><h2>Top {{ candidates|length }} Matches</h2><div style="overflow-x:auto;"><table><thead><tr><th>Rank</th><th>Candidate ID</th><th>Filename</th><th>Department</th><th>Score</th><th>Explanation</th><th>Preview</th></tr></thead><tbody>{% for c in candidates %}<tr><td>{{ loop.index }}</td><td>{{ c.candidate_id }}</td><td>{{ c.filename }}</td><td>{{ c.department }}</td><td class="score-high">{{ c.score }}%</td><td>{{ c.explanation }}</td><td><pre>{{ c.text_preview }}</pre></td></tr>{% endfor %}</tbody></table></div></div>{% endif %}</div></body></html>
'''

SESSION_VIEW_TEMPLATE = '''
<!DOCTYPE html><html><head><title>Session Details</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><style>body{font-family:'Inter',sans-serif;background:#f0f2f5;padding:40px;}.container{max-width:1200px;margin:0 auto;}.card{background:white;border-radius:28px;padding:24px;margin-bottom:24px;}table{width:100%;border-collapse:collapse;}th,td{padding:12px;border-bottom:1px solid #e2e8f0;}.score-high{color:#10b981;font-weight:bold;}pre{background:#f8fafc;padding:8px;border-radius:12px;}</style></head><body><div class="container"><div class="card"><h2>Session #{{ session_id }}</h2><p><strong>Time:</strong> {{ timestamp[:19] }}</p><p><strong>Department:</strong> {{ department }}</p><p><strong>Job Descriptions:</strong></p><pre>{{ job_descs }}</pre></div><div class="card"><h2>Candidates</h2><div style="overflow-x:auto;"><table><thead><tr><th>Candidate ID</th><th>Filename</th><th>Best Job</th><th>Score</th><th>Score Details</th><th>Preview</th></tr></thead><tbody>{% for c in candidates %}<tr><td>{{ c.candidate_id }}</td><td>{{ c.filename }}</td><td>{{ c.best_job }}</td><td class="score-high">{{ c.best_score }}%</td><td>Skills:{{ c.details.skills_score }} Exp:{{ c.details.experience_score }} Edu:{{ c.details.education_score }} Cert:{{ c.details.certificate_score }}</td><td><pre>{{ c.text_preview }}</pre></td></tr>{% endfor %}</tbody></table></div></div></div></body></html>
'''

# ---------- Routes ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT company_id FROM users WHERE username=? AND password=?", (username, password))
        row = c.fetchone()
        conn.close()
        if row:
            session['company_id'] = row[0]
            return redirect(url_for('index'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials")
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name FROM companies")
        companies = [{'id': row[0], 'name': row[1]} for row in c.fetchall()]
        conn.close()
        return render_template_string(REGISTER_TEMPLATE, companies=companies, error=None)
    else:
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        company_id = request.form['company_id']
        if not username or not password or not company_id:
            return render_template_string(REGISTER_TEMPLATE, companies=[], error="All fields required.")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Check username existence
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return render_template_string(REGISTER_TEMPLATE, companies=[], error="Username already exists.")
        # Insert new user
        c.execute("INSERT INTO users (username, password, company_id) VALUES (?, ?, ?)", (username, password, company_id))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('company_id', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM companies WHERE id=?", (company_id,))
    company_name = c.fetchone()[0]
    conn.close()
    return render_template_string(MAIN_TEMPLATE, candidates=None, error=None, company_name=company_name)

@app.route('/upload', methods=['POST'])
def upload():
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    file = request.files.get('file')
    candidate_id = request.form.get('candidate_id', '').strip()
    department = request.form.get('department', 'technology')
    job_descs_raw = request.form.get('job_descs', '')
    if not file or not candidate_id:
        return render_template_string(MAIN_TEMPLATE, error="Candidate ID and file required.", candidates=None, company_name="")
    if not allowed_file(file.filename):
        return render_template_string(MAIN_TEMPLATE, error="File type not allowed.", candidates=None, company_name="")
    if not job_descs_raw.strip():
        return render_template_string(MAIN_TEMPLATE, error="Please enter job descriptions.", candidates=None, company_name="")
    job_descs = [line.strip() for line in job_descs_raw.split('\n') if line.strip()]
    if not job_descs:
        return render_template_string(MAIN_TEMPLATE, error="No valid job descriptions.", candidates=None, company_name="")
    
    # Insert session
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO sessions (company_id, timestamp, job_descs, department) VALUES (?, ?, ?, ?)",
              (company_id, timestamp, json.dumps(job_descs), department))
    session_id = c.lastrowid
    conn.commit()
    
    # Process file
    filename = file.filename
    file_bytes = file.read()
    text = get_text_from_file_bytes(filename, file_bytes)
    if not text or len(text.strip()) < 50:
        conn.close()
        return render_template_string(MAIN_TEMPLATE, error="Could not extract text from file.", candidates=None, company_name="")
    
    # Add/overwrite in library
    add_to_library(company_id, candidate_id, filename, department, text)
    text_preview = text[:400] + ('...' if len(text) > 400 else '')
    
    best_score = -1
    best_job = ""
    best_details = None
    for job_desc in job_descs:
        score, details = match_with_ollama_detailed(text, job_desc, department)
        if score > best_score:
            best_score = score
            best_job = job_desc[:100]
            best_details = details
    if best_score == -1:
        conn.close()
        return render_template_string(MAIN_TEMPLATE, error="Matching failed.", candidates=None, company_name="")
    
    encrypted_text = encrypt_text(text)
    encrypted_preview = encrypt_text(text_preview)
    c.execute('''INSERT INTO candidates (session_id, company_id, candidate_id, department, filename, extracted_text, best_score, best_job, explanation, text_preview, score_details)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (session_id, company_id, candidate_id, department, filename, encrypted_text, best_score, best_job, json.dumps(best_details), encrypted_preview, json.dumps(best_details)))
    conn.commit()
    conn.close()
    
    candidates = [{
        'candidate_id': candidate_id,
        'filename': filename,
        'best_job': best_job,
        'score': best_score,
        'text_preview': text_preview,
        'details': best_details
    }]
    candidates.sort(key=lambda x: x['score'], reverse=True)
    conn2 = sqlite3.connect(DB_PATH)
    c2 = conn2.cursor()
    c2.execute("SELECT name FROM companies WHERE id=?", (company_id,))
    company_name = c2.fetchone()[0]
    conn2.close()
    return render_template_string(MAIN_TEMPLATE, candidates=candidates[:10], error=None, company_name=company_name)

@app.route('/test_huggingface_dataset', methods=['POST'])
def test_huggingface_dataset():
    if not is_logged_in():
        return redirect(url_for('login'))
    try:
        num_resumes = min(int(request.form.get('num_resumes', 5)), 10)
        num_jobs = min(int(request.form.get('num_jobs', 3)), 5)
    except:
        num_resumes, num_jobs = 5, 3
    try:
        ds = load_dataset("cnamuangtoun/resume-job-description-fit", split="train")
    except Exception as e:
        return render_template_string(MAIN_TEMPLATE, candidates=None, error=f"Dataset error: {e}", company_name="")
    df = ds.to_pandas()
    resumes = df['resume_text'].tolist()[:num_resumes]
    unique_jobs = df['job_description_text'].unique().tolist()
    job_descs = unique_jobs[:num_jobs]
    if not resumes or not job_descs:
        return render_template_string(MAIN_TEMPLATE, candidates=None, error="Not enough data", company_name="")
    candidates = []
    for idx, text in enumerate(resumes):
        best_score = -1
        best_job = ""
        best_details = None
        for job in job_descs:
            score, details = match_with_ollama_detailed(text, job, "technology")
            if score > best_score:
                best_score = score
                best_job = job[:100]
                best_details = details
        if best_score == -1:
            continue
        candidates.append({
            'candidate_id': f"Sample_{idx+1}",
            'filename': f"HuggingFace Sample {idx+1}",
            'best_job': best_job,
            'score': best_score,
            'text_preview': text[:300]+'...',
            'details': best_details
        })
    if not candidates:
        return render_template_string(MAIN_TEMPLATE, candidates=None, error="No matches", company_name="")
    candidates.sort(key=lambda x: x['score'], reverse=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM companies WHERE id=?", (get_current_company_id(),))
    company_name = c.fetchone()[0] if c.fetchone() else "Company"
    conn.close()
    return render_template_string(MAIN_TEMPLATE, candidates=candidates[:10], error=None, company_name=company_name)

@app.route('/history')
def history():
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, department, job_descs FROM sessions WHERE company_id=? ORDER BY id DESC", (company_id,))
    rows = c.fetchall()
    conn.close()
    sessions = [{'id': r[0], 'timestamp': r[1], 'department': r[2], 'job_descs': r[3]} for r in rows]
    return render_template_string(HISTORY_TEMPLATE, sessions=sessions)

@app.route('/library')
def library():
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    dept_filter = request.args.get('dept_filter', 'all')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if dept_filter == 'all':
        c.execute("SELECT id, candidate_id, filename, department, upload_time, text_preview FROM resume_library WHERE company_id=? ORDER BY upload_time DESC", (company_id,))
    else:
        c.execute("SELECT id, candidate_id, filename, department, upload_time, text_preview FROM resume_library WHERE company_id=? AND department=? ORDER BY upload_time DESC", (company_id, dept_filter))
    rows = c.fetchall()
    conn.close()
    resumes = []
    for row in rows:
        resumes.append({'id': row[0], 'candidate_id': row[1], 'filename': row[2], 'department': row[3], 'upload_time': row[4], 'text_preview': decrypt_text(row[5])})
    return render_template_string(LIBRARY_TEMPLATE, resumes=resumes)

@app.route('/delete_from_library/<int:resume_id>')
def delete_from_library(resume_id):
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM resume_library WHERE id=? AND company_id=?", (resume_id, company_id))
    conn.commit()
    conn.close()
    return redirect(url_for('library'))

@app.route('/match_from_library', methods=['GET', 'POST'])
def match_from_library():
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    if request.method == 'GET':
        return render_template_string(MATCH_TEMPLATE, candidates=None)
    job_desc = request.form.get('job_desc', '').strip()
    dept = request.form.get('dept', 'all')
    if not job_desc:
        return render_template_string(MATCH_TEMPLATE, candidates=None, error="Please enter job description.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if dept == 'all':
        c.execute("SELECT candidate_id, filename, department, extracted_text, text_preview FROM resume_library WHERE company_id=?", (company_id,))
    else:
        c.execute("SELECT candidate_id, filename, department, extracted_text, text_preview FROM resume_library WHERE company_id=? AND department=?", (company_id, dept))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return render_template_string(MATCH_TEMPLATE, candidates=None, error="No resumes found in this department.")
    candidates = []
    for candidate_id, filename, department, enc_text, enc_preview in rows:
        dec_text = decrypt_text(enc_text)
        dec_preview = decrypt_text(enc_preview)
        score, explanation = match_with_ollama_simple(dec_text, job_desc)
        candidates.append({'candidate_id': candidate_id, 'filename': filename, 'department': department, 'score': score, 'explanation': explanation, 'text_preview': dec_preview})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return render_template_string(MATCH_TEMPLATE, candidates=candidates[:10])

@app.route('/view_session/<int:session_id>')
def view_session(session_id):
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, department, job_descs FROM sessions WHERE id=? AND company_id=?", (session_id, company_id))
    session_row = c.fetchone()
    if not session_row:
        conn.close()
        return "Session not found", 404
    timestamp, department, job_descs_json = session_row
    job_descs = json.loads(job_descs_json)
    c.execute("SELECT candidate_id, filename, best_score, best_job, explanation, text_preview, score_details FROM candidates WHERE session_id=? AND company_id=? ORDER BY best_score DESC", (session_id, company_id))
    rows = c.fetchall()
    conn.close()
    candidates = []
    for row in rows:
        candidates.append({
            'candidate_id': row[0],
            'filename': row[1],
            'best_score': row[2],
            'best_job': row[3],
            'explanation': row[4],
            'text_preview': decrypt_text(row[5]),
            'details': json.loads(row[6]) if row[6] else {}
        })
    return render_template_string(SESSION_VIEW_TEMPLATE, session_id=session_id, timestamp=timestamp, department=department, job_descs="\n".join(job_descs), candidates=candidates)

@app.route('/delete_session/<int:session_id>')
def delete_session(session_id):
    if not is_logged_in():
        return redirect(url_for('login'))
    company_id = get_current_company_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id=? AND company_id=?", (session_id, company_id))
    conn.commit()
    conn.close()
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)