import streamlit as st
from core.nlp import ResumeRanker
import pdfplumber
import os
from pathlib import Path
import tempfile

def read_pdf(file):
    """Extract text from PDF file"""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        raise Exception(f"Error processing {file.name if hasattr(file, 'name') else 'PDF file'}: {str(e)}")

def main():
    st.title("Resume Ranking System")
    
    # Initialize resume ranker
    ranker = ResumeRanker()
    
    # File uploader for resumes
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF resumes",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    # Job description input
    st.header("Enter Job Description")
    job_description = st.text_area(
        "Job Description",
        "Enter the job description here...",
        height=200
    )
    
    # Process button
    if st.button("Rank Resumes"):
        if not uploaded_files:
            st.warning("Please upload at least one resume")
            return
            
        if not job_description.strip():
            st.warning("Please enter a job description")
            return
            
        # Process resumes
        resume_texts = []
        for file in uploaded_files:
            try:
                # First, check if the file is a valid PDF by checking the magic number
                if file.getvalue()[:4] != b'%PDF':
                    st.error(f"{file.name} is not a valid PDF file (missing PDF header)")
                    continue
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    resume_text = read_pdf(tmp_file_path)
                    if not resume_text.strip():
                        st.warning(f"No text could be extracted from {file.name}. The PDF might be image-based or encrypted.")
                        continue
                    resume_texts.append({
                        'name': file.name,
                        'text': resume_text
                    })
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
                finally:
                    try:
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                    except Exception:
                        pass
                        
            except Exception as e:
                st.error(f"Error handling {file.name}: {str(e)}")
                continue
        
        # Rank resumes
        ranked_resumes = ranker.rank_resumes(job_description, resume_texts)
        
        # Display results
        st.header("Ranked Resumes")
        if not ranked_resumes:
            st.warning("No valid resumes could be processed for ranking.")
            return
            
        for idx, resume in enumerate(ranked_resumes):
            with st.expander(f"Rank {idx + 1} - {resume['name']} (Score: {resume['similarity_score']:.2f})"):
                st.subheader("Resume Content:")
                st.text(resume['resume_text'][:1000] + ("..." if len(resume['resume_text']) > 1000 else ""))
                st.caption(f"Showing first 1000 characters of {len(resume['resume_text'])} total characters")

if __name__ == "__main__":
    main()
