import os
import pickle
import json
import re
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Legal Text Processor (simplified version for web deployment)
class WebLegalTextProcessor:
    def __init__(self):
        self.legal_patterns = {
            'acts': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act,?\s*(\d{4})\b',
            'sections': r'\b[Ss]ection\s+(\d+[A-Za-z]*)\b',
            'government_bodies': r'\b(Central\s+Government|State\s+Government|Central\s+Board|Tribunal)\b',
            'percentages': r'\b(\d+(?:\.\d+)?)\s*(?:per\s*cent|%)\b',
            'obligations': r'\b(shall|must|required\s+to|obligated\s+to)\s+([^.]+)',
        }
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'section\s+\d+[a-z]*', 'section', text)
        text = re.sub(r'clause\s+\d+[a-z]*', 'clause', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        return text.strip()
    
    def extract_information(self, text):
        extracted_info = {}
        for category, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    cleaned_matches = list(set([' '.join(match) if isinstance(match, tuple) else match for match in matches]))
                else:
                    cleaned_matches = list(set(matches))
                extracted_info[category] = cleaned_matches[:3]
        return extracted_info
    
    def classify_text(self, text):
        # Simplified classification based on keywords
        text_lower = text.lower()
        scores = {
            'EPF': sum(1 for word in ['provident', 'fund', 'employee', 'contribution', 'epf'] if word in text_lower),
            'Companies': sum(1 for word in ['company', 'corporation', 'limited', 'memorandum', 'directors'] if word in text_lower),
            'TDS': sum(1 for word in ['tax', 'deducted', 'source', 'tds', 'income'] if word in text_lower),
            'Contract': sum(1 for word in ['agreement', 'contract', 'parties', 'terms', 'breach'] if word in text_lower),
            'Regulation': sum(1 for word in ['regulation', 'rule', 'compliance', 'authority'] if word in text_lower)
        }
        
        predicted_category = max(scores.keys(), key=lambda k: scores[k])
        max_score = max(scores.values())
        confidence = max_score / (sum(scores.values()) + 1)  # Normalize
        
        return {
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'all_scores': {k: float(v) for k, v in scores.items()}
        }

# Initialize processor
processor = WebLegalTextProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze the text
        classification = processor.classify_text(text)
        entities = processor.extract_information(text)
        preprocessed = processor.preprocess_text(text)
        
        # Create simple summary (first 2 sentences)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        summary = sentences[:2]
        
        result = {
            'classification': classification,
            'entities': entities,
            'summary': summary,
            'preprocessed_text': preprocessed[:300] + "..." if len(preprocessed) > 300 else preprocessed,
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'model': 'Legal Text Processor v1.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)