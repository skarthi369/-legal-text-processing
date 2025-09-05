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

app = Flask(__name__, 
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')))

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
        # Enhanced classification with weighted keywords
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Define keywords with weights
        keyword_weights = {
            'EPF': {
                'provident': 2.0, 'fund': 1.5, 'employee': 1.0, 'contribution': 1.5, 'epf': 2.0,
                'pension': 1.5, 'employer': 1.0, 'establishment': 1.0, 'scheme': 1.0,
                'withdrawal': 1.0, 'universal': 1.0, 'account': 1.0
            },
            'Companies': {
                'company': 2.0, 'corporation': 1.5, 'limited': 1.5, 'memorandum': 1.5, 'directors': 1.5,
                'board': 1.0, 'shareholder': 1.5, 'articles': 1.0, 'incorporation': 1.5,
                'register': 1.0, 'resolution': 1.0, 'secretary': 1.0
            },
            'TDS': {
                'tax': 2.0, 'deducted': 2.0, 'source': 1.5, 'tds': 2.0, 'income': 1.5,
                'deduction': 1.5, 'withholding': 1.5, 'assessment': 1.0, 'return': 1.0,
                'payment': 1.0, 'certificate': 1.0, 'challan': 1.0
            },
            'Contract': {
                'agreement': 2.0, 'contract': 2.0, 'parties': 1.5, 'terms': 1.5, 'breach': 1.5,
                'clause': 1.0, 'covenant': 1.5, 'obligation': 1.5, 'execution': 1.0,
                'consideration': 1.5, 'termination': 1.5, 'liability': 1.0
            },
            'Regulation': {
                'regulation': 2.0, 'rule': 1.5, 'compliance': 1.5, 'authority': 1.5, 'statutory': 1.5,
                'notification': 1.0, 'provision': 1.0, 'gazette': 1.0, 'enforcement': 1.0,
                'guidelines': 1.0, 'directive': 1.0, 'amendment': 1.0
            }
        }
        
        # Calculate weighted scores
        scores = {}
        for category, keywords in keyword_weights.items():
            category_score = sum(weight for word, weight in keywords.items() if word in text_lower)
            # Normalize by maximum possible score for the category
            max_possible = sum(keywords.values())
            scores[category] = category_score / max_possible if max_possible > 0 else 0
        
        predicted_category = max(scores.keys(), key=lambda k: scores[k])
        max_score = max(scores.values())
        
        # Enhanced confidence calculation
        confidence = max_score * (1 + np.log1p(len(text_words)) / 100)  # Scale with text length
        confidence = min(max(confidence, 0.1), 0.99)  # Bound between 0.1 and 0.99
        
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