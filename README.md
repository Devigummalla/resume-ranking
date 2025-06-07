
# resume-ranking
# Resume Ranking System

A web application that ranks resumes based on their relevance to a job description using NLP techniques.

## Features

- Upload multiple PDF resumes
- Enter job description
- Rank resumes based on semantic similarity
- View ranked results with similarity scores
- Preview resume content

## Requirements

- Python 3.8+
- pip

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (usually http://localhost:8501)

3. Upload PDF resumes and enter a job description

4. Click "Rank Resumes" to see the ranked results

## Technologies Used

- Streamlit: For the web interface
- Sentence Transformers: For text embeddings and similarity calculation
- PDFPlumber: For PDF text extraction
- Python: Backend processing
d4eab89 (created resume ranking system with nlp)
