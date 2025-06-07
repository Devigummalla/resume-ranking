from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Tuple
import torch

class ResumeRanker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the ResumeRanker with a pre-trained sentence transformer model.
        
        Args:
            model_name: Name of the pre-trained model to use. Default is 'all-MiniLM-L6-v2'.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra whitespace
        return text.strip()
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for the input text.
        
        Args:
            text: Input text to create embeddings for.
            
        Returns:
            Numpy array containing the text embeddings.
        """
        if not text.strip():
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
            
        text = self.preprocess_text(text)
        with torch.no_grad():
            return self.model.encode(text, convert_to_numpy=True)
    
    def rank_resumes(self, job_description: str, resume_texts: List[Dict]) -> List[Dict]:
        """Rank resumes based on similarity to job description.
        
        Args:
            job_description: Job description text.
            resume_texts: List of dictionaries containing resume information.
                         Each dict should have at least a 'text' key with the resume content.
                         
        Returns:
            List of dictionaries containing ranked resumes with similarity scores.
        """
        if not job_description.strip() or not resume_texts:
            return []
        
        # Get embeddings
        job_desc_embedding = self.get_embeddings(job_description)
        
        # Process resumes
        results = []
        for idx, resume in enumerate(resume_texts):
            try:
                resume_text = resume.get('text', '')
                if not resume_text.strip():
                    continue
                    
                resume_embedding = self.get_embeddings(resume_text)
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(
                    torch.tensor(job_desc_embedding).unsqueeze(0),
                    torch.tensor(resume_embedding).unsqueeze(0)
                ).item()
                
                results.append({
                    'resume_idx': idx,
                    'similarity_score': similarity,
                    'resume_text': resume_text,
                    'name': resume.get('name', f'Resume {idx + 1}')
                })
                
            except Exception as e:
                print(f"Error processing resume {idx}: {str(e)}")
                continue
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
