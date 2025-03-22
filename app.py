import os
import PyPDF2
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def rank_resumes(job_description, resume_texts):
    """Rank resumes based on their similarity to the job description."""
    if not resume_texts:
        return []
    texts = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        job_description = request.form['job_description']
        files = request.files.getlist('files')
        resume_texts = []
        filenames = []
        
        for file in files:
            if file.filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath)
                if text:
                    resume_texts.append(text)
                    filenames.append(file.filename)
        
        rankings = rank_resumes(job_description, resume_texts)
        ranked_resumes = [(filenames[i], score) for i, score in rankings]
        
        return render_template('index.html', ranked_resumes=ranked_resumes)
    
    return render_template('index.html', ranked_resumes=None)

if __name__ == '__main__':
    app.run(debug=True)