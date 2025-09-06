import os
import logging
import chromadb
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving legal documents"""
    
    def __init__(self, db_path: str = "./data/embeddings", collection_name: str = "legal_documents"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        logger.info(f"Vector store initialized with {self.get_document_count()} documents")

    def add_document(self, text: str, source: str, metadata: Optional[Dict] = None) -> str:
        """Add document to vector store"""
        try:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Split text into chunks
            chunks = self._split_text(text)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            base_metadata = {
                'source': source,
                'doc_id': doc_id,
                **(metadata or {})
            }
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk_metadata = {
                    **base_metadata,
                    'chunk_index': i,
                    'chunk_id': chunk_id
                }
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added document {source} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') else None,
                        'score': 1 - results['distances'][0][i] if results.get('distances') else 1.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            return len(self.list_documents())
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store"""
        try:
            # Get all documents
            results = self.collection.get()
            
            # Group by doc_id
            doc_groups = {}
            for i, metadata in enumerate(results['metadatas']):
                doc_id = metadata.get('doc_id')
                if doc_id:
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = {
                            'doc_id': doc_id,
                            'source': metadata.get('source', 'Unknown'),
                            'chunks': 0,
                            'metadata': metadata
                        }
                    doc_groups[doc_id]['chunks'] += 1
            
            return list(doc_groups.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def load_documents(self):
        """Load pre-created legal documents"""
        try:
            # Check if we already have documents
            if self.get_document_count() > 0:
                logger.info("Documents already loaded")
                return

            # Sample legal documents
            legal_documents = {
                "EPF Act 1952": """
                THE EMPLOYEES' PROVIDENT FUNDS AND MISCELLANEOUS PROVISIONS ACT, 1952

                Section 1. Short title, extent and application.
                (1) This Act may be called the Employees' Provident Funds and Miscellaneous Provisions Act, 1952.
                (2) It extends to the whole of India.
                (3) It applies to every establishment which is a factory engaged in any industry specified in Schedule I and in which twenty or more persons are employed.

                Section 6. Contributions.
                The contribution which shall be paid by the employer to the Fund shall be ten per cent of the basic wages, dearness allowance and retaining allowance for each employee. The employees' contribution shall be equal to the contribution payable by the employer.

                Section 154. Information in cognizable cases.
                How to file complaint against SHO: Write application referencing CrPC Section 154(3), attach copy of FIR refusal notice, submit to Superintendent of Police office.
                """,

                "Companies Act 2013": """
                THE COMPANIES ACT, 2013

                Section 128. Books of account.
                (1) Every company shall keep at its registered office proper books of account.
                (2) The books of account shall give a true and fair view of the state of affairs of the company.

                Section 96. Annual general meeting.
                (1) Every company shall hold an annual general meeting within six months of the financial year end.

                Audit requirements: Companies must appoint auditors and conduct annual audits as per Section 139-148.
                """,

                "TDS Guidelines": """
                TAX DEDUCTED AT SOURCE (TDS)

                Section 192. Salary TDS.
                Tax shall be deducted at source from salary payments at applicable rates.

                TDS Rates:
                - Salary: As per income tax slab rates
                - Interest: 10% if exceeding Rs. 40,000
                - Professional fees: 10% if exceeding Rs. 30,000

                Form 26AS shows TDS details for taxpayer verification.
                """
            }

            for title, content in legal_documents.items():
                self.add_document(
                    text=content,
                    source=title,
                    metadata={'type': 'legal_act'}
                )
                logger.info(f"Loaded {title}")

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")

    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start < 0:
                start = 0
        
        return chunks
