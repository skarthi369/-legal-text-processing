import os
import re
import logging
import pdfplumber
import PyPDF2
from io import BytesIO
from typing import Dict, List, Any
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor for legal texts"""
    
    def __init__(self):
        self.legal_keywords = {
            'EPF': [
                'provident fund', 'epf', 'employee provident fund', 'pension fund',
                'contribution', 'basic wages', 'dearness allowance'
            ],
            'Companies': [
                'company', 'companies act', 'memorandum', 'board of directors',
                'shareholders', 'audit', 'annual general meeting'
            ],
            'TDS': [
                'tax deducted at source', 'tds', 'income tax', 'deduction',
                'form 16', 'form 26as', 'withholding tax'
            ],
            'Contract': [
                'agreement', 'contract', 'parties', 'consideration', 'breach',
                'damages', 'terms and conditions'
            ],
            'Criminal': [
                'indian penal code', 'ipc', 'criminal procedure code', 'crpc',
                'fir', 'police', 'magistrate', 'arrest'
            ],
            'Civil': [
                'civil procedure code', 'cpc', 'civil court', 'suit',
                'plaintiff', 'defendant'
            ]
        }
        
        self.legal_patterns = {
            'acts': r'\b([A-Z][a-zA-Z\s]+)\s+Act,?\s*(\d{4})\b',
            'sections': r'\b[Ss]ection\s+(\d+[A-Za-z]*)\b',
            'amounts': r'\b(?:Rs\.?\s*|rupees\s+)?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:rupees|crores?|lakhs?)?\b',
            'percentages': r'\b(\d+(?:\.\d+)?)\s*(?:per\s*cent|%)\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        }
        
        logger.info("Document processor initialized")

    def process_uploaded_file(self, file: FileStorage) -> str:
        """Process uploaded file and extract text"""
        try:
            filename = file.filename.lower()
            
            if filename.endswith('.pdf'):
                return self._extract_pdf_text(file)
            elif filename.endswith('.txt'):
                return self._extract_txt_text(file)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise

    def _extract_pdf_text(self, file: FileStorage) -> str:
        """Extract text from PDF file"""
        try:
            file_bytes = BytesIO(file.read())
            text_content = []
            
            # Try pdfplumber first
            with pdfplumber.open(file_bytes) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            if text_content:
                return self.clean_text("\n".join(text_content))
            
            # Fallback to PyPDF2
            file.seek(0)
            file_bytes = BytesIO(file.read())
            reader = PyPDF2.PdfReader(file_bytes)
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
            
            return self.clean_text("\n".join(text_content))
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def _extract_txt_text(self, file: FileStorage) -> str:
        """Extract text from TXT file"""
        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return self.clean_text(content)
        except Exception as e:
            logger.error(f"Error extracting TXT text: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess legal text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal formatting
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\'\"\/]', ' ', text)
        
        # Normalize legal references
        text = re.sub(r'\bSec\b\.?\s*', 'Section ', text)
        text = re.sub(r'\bArt\b\.?\s*', 'Article ', text)
        
        return text.strip()

    def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify legal document based on keywords"""
        try:
            text_lower = text.lower()
            scores = {}
            
            for category, keywords in self.legal_keywords.items():
                score = 0
                found_keywords = []
                
                for keyword in keywords:
                    count = text_lower.count(keyword.lower())
                    if count > 0:
                        score += count
                        found_keywords.append(keyword)
                
                scores[category] = {
                    'score': score,
                    'keywords_found': found_keywords
                }
            
            # Find best match
            best_category = max(scores.keys(), key=lambda k: scores[k]['score'])
            total_score = sum(score['score'] for score in scores.values())
            confidence = scores[best_category]['score'] / total_score if total_score > 0 else 0
            
            return {
                'predicted_category': best_category,
                'confidence': round(confidence, 4),
                'all_scores': {cat: data['score'] for cat, data in scores.items()},
                'keywords_found': scores[best_category]['keywords_found'],
                'method': 'keyword_based'
            }
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return {
                'predicted_category': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }

    def extract_information(self, text: str) -> Dict[str, Any]:
        """Extract legal entities from text"""
        try:
            extracted_info = {}
            
            for entity_type, pattern in self.legal_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                if matches:
                    # Clean matches
                    if matches and isinstance(matches[0], tuple):
                        cleaned_matches = []
                        for match in matches:
                            if isinstance(match, tuple):
                                if len(match) > 1:
                                    cleaned_matches.append(f"{match[0]} {match[1]}" if match[1] else match[0])
                                else:
                                    cleaned_matches.append(match[0])
                            else:
                                cleaned_matches.append(match)
                    else:
                        cleaned_matches = list(matches)
                    
                    # Remove duplicates
                    unique_matches = list(dict.fromkeys(cleaned_matches))[:5]
                    
                    extracted_info[entity_type] = {
                        'matches': unique_matches,
                        'count': len(matches),
                        'description': self._get_entity_description(entity_type)
                    }
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return {'error': str(e)}

    def _get_entity_description(self, entity_type: str) -> str:
        """Get description for entity type"""
        descriptions = {
            'acts': 'Legal Acts and Laws',
            'sections': 'Legal Sections',
            'amounts': 'Monetary Amounts',
            'percentages': 'Percentage Values',
            'dates': 'Dates and Years'
        }
        return descriptions.get(entity_type, 'Legal Entities')
