import os
import time
import logging
from typing import List, Dict, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiRAG:
    """Gemini-based RAG system for legal documents"""
    
    def __init__(self, vector_store, model_name: str = "gemini-pro"):
        self.vector_store = vector_store
        self.model_name = model_name
        
        # Configure Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini RAG with model: {model_name}")

    def get_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get response to user query using RAG"""
        start_time = time.time()
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.search(query, top_k=top_k)
            
            # Create context from retrieved documents
            context = self._create_context(relevant_docs)
            
            # Generate response using Gemini
            prompt = self._create_prompt(query, context)
            response = self.model.generate_content(prompt)
            
            processing_time = time.time() - start_time
            
            return {
                'answer': response.text,
                'sources': [doc['metadata'] for doc in relevant_docs],
                'context_used': len(relevant_docs),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'context_used': 0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def _create_context(self, relevant_docs: List[Dict]) -> str:
        """Create context string from relevant documents"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"Document {i+1}:")
            context_parts.append(f"Source: {doc['metadata'].get('source', 'Unknown')}")
            context_parts.append(f"Content: {doc['content']}")
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini"""
        return f"""You are an expert legal assistant specializing in Indian law. Use the provided context to answer the user's question accurately.

Context from Legal Documents:
{context}

User Question: {query}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain relevant information, clearly state this
3. Provide specific legal provisions, section numbers, and act names when applicable
4. Give practical, actionable advice when appropriate
5. Use clear, professional language
6. If referring to legal procedures, provide step-by-step guidance

Please provide a detailed and helpful response:"""

    def summarize_text(self, text: str) -> str:
        """Summarize legal text using Gemini"""
        try:
            prompt = f"""Summarize the following legal text clearly and concisely:

{text}

Provide a summary that captures the essential legal points:"""

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return "Unable to generate summary at this time."
