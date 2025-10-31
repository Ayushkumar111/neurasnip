""" 
Vector database using FAISS 
handles storage , indexing , and similiarty search  of image embeddings

"""

import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List , Dict , Tuple , Optional
from loguru import logger


#configure logger 

logger.add("data/logs/vector_db.log", rotation="10 MB")

class VectorDatabase:
      
    def __init__(
        self,
        index_path: str = "data/vector_store/faiss_index.bin",
        metadata_path: str = "data/vector_store/metadata.pkl",
        dimension: int = 512
    ):
        
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path

        # initlizing faiss index flatip for consine similarity of normalized vectors
        self.index = faiss.IndexFlatIP(dimension)

        #metadata storage using dict as faiss only stores vectors
        self.metadata =[]

        self.next_id =0

        logger.info(f"VectorDatabase initialized with dimension: {dimension}")
        logger.info(f"Index path: {index_path}")
        logger.info(f"Metadata path: {metadata_path}")


    def add_vector(
        self,
        vector: np.ndarray,
        metadata: Dict
    ) -> int:
        
        """ we will add a single vector with metadata to the database
        arg - vector: Embedding vector (shape: (512,))
            metadata: Dict with info like:
                {
                    'filename': 'receipt.jpg',
                    'path': 'D:/images/receipt.jpg',
                    'ocr_text': 'COFFEE SHOP...',
                    'hash': 'abc123'
                }

        return - a unique vector id that would be assigned to this vector         
        """

        try:
            if vector.shape !=(self.dimension,):
                logger.error(f"Vector shape mismatch. Expected: {(self.dimension,)}, Got: {vector.shape}")
                return -1
            vector_2d = vector.reshape(1, -1).astype('float32') # faiss expect batch format 2d array even for a single vector
            self.index.add(vector_2d)

            vector_id = self.next_id
            metadata['vector_id'] = vector_id
            self.metadata.append(metadata)

            self.next_id +=1

            logger.debug(f"Added vector ID {vector_id} with metadata: {metadata}")
            return vector_id
        
        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            return -1


    def add_vectors_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict]
    ) -> List[int]:
        """
        Add multiple vectors at once (faster!)
        
        Args:
            vectors: Array of shape (N, 512) - N vectors
            metadata_list: List of N metadata dicts
            
        Returns:
            List of assigned vector IDs
        """
        try:
            # validate first as we are adding all at once
            if vectors.shape[0] != len(metadata_list):
                logger.error("Vectors and metadata count mismatch")
                return []
            
            if vectors.shape[1] != self.dimension:
                logger.error(f"Invalid dimension: {vectors.shape[1]}")
                return []
            
            # convert to float32 (FAISS requirement)
            vectors = vectors.astype('float32')
            
            # add all vectors at once
            # Why faster: Single FAISS call instead of N calls
            self.index.add(vectors)
            
            # Assign IDs and store metadata
            vector_ids = []
            for metadata in metadata_list:
                vector_id = self.next_id
                metadata['vector_id'] = vector_id
                self.metadata.append(metadata)
                vector_ids.append(vector_id)
                self.next_id += 1
            
            logger.info(f" Added batch of {len(vectors)} vectors")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return [] 

    def search(
        self,
        query_vector: np.ndarray,
        top_k:int = 5
    )-> List[Dict]:
        """ search for similar vectors in db
        Args:
            query_vector: Vector to search for (512,)
            top_k: Return top K most similar results
            
        Returns:
            List of dicts with results:
            [
                {
                    'vector_id': 0,
                    'similarity': 0.89,
                    'filename': 'receipt.jpg',
                    'path': 'D:/images/receipt.jpg',
                    'ocr_text': '...'
                },
                ...
            ]
        """  

        try:
            # checking if db is empty
            if self.index.ntotal ==0:
                logger.warning("Search attempted on empty database")
                return []

            top_k = min(top_k , self.index.ntotal)

            query_2d = query_vector.reshape(1,-1).astype('float32')

            similarities , indices = self.index.search(query_2d , top_k)

            similarities = similarities[0]
            indices = indices[0] # removing batch dimensions

            # now building results with metadata
            results =[]
            for idx, similarity in zip(indices , similarities):
                if idx<0 or idx >= len(self.metadata):
                    logger.warning(f"Invalid index returned: {idx}")
                    continue 
                result = self.metadata[idx].copy()
                result['similarity'] = float(similarity)
                results.append(result)

            logger.info(f"Search completed. Top {len(results)} results returned.")    
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_text(
        self,
        text_query: str,
        embedder,
        top_k: int = 5
    ) -> List[Dict]:
        """ search using text query 
        User will type a text query , we will generate embedding and search and find matching images

        Args:
            text_query: Natural language query
            embedder: CLIPEmbeddings instance to convert text to vector
            top_k: Number of results
            
        Returns:
            List of matching images with metadata

        """
        try:
            query_vector = embedder.generate_text_embedding(text_query)

            results = self.search(query_vector , top_k)

            logger.info(f"Text search: '{text_query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
        

        # will check text embeding import once if issue aarives

    def save(self) -> bool:
        """ save index and metadata to disk as we dont re index on every run """        

        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            faiss.write_index(self.index, self.index_path)

            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'next_id': self.next_id
                }, f)

            logger.info(f"Vector database saved to {self.index_path} and {self.metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            return False
        
    def load(self) -> bool:
        """ load index and metadata from disk 
        we will call this function on app startup to restore previous state
        
        """

        try:
            if not os.path.exists(self.index_path):
                logger.warning(f"Index file not found: {self.index_path}")
                return False
            
            self.index = faiss.read_index(self.index_path)

            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', [])
                self.next_id = data.get('next_id', len(self.metadata))

            logger.info(f"Vector database loaded from {self.index_path} and {self.metadata_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return False

    def remove_by_hash(self, file_hash:str) -> bool:
        """ remove vector by file hash in case of duplicate because faiss do not support effeicnt deletion
        so mark as deleted , and rebuild the inex later"""

        try:
            # find metadata entry
            for i, meta in enumerate(self.metadata):
                if meta.get('hash') == file_hash:
                    # mark as deleted (we'll skip in search results)
                    self.metadata[i]['deleted'] = True
                    logger.info(f"🗑️  Marked vector {i} as deleted")
                    return True
            
            logger.warning(f"Hash not found: {file_hash}")
            return False
            
        except Exception as e:
            logger.error(f"Remove failed: {e}")
            return False
        
    def get_stats(self) -> Dict:
        """Get database statistics"""
        active_count = sum(1 for m in self.metadata if not m.get('deleted', False))
        
        return {
            'total_vectors': self.index.ntotal,
            'active_vectors': active_count,
            'deleted_vectors': len(self.metadata) - active_count,
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP',
            'metadata_entries': len(self.metadata)
        }
    
    
    def clear(self):
        """Clear all data (useful for testing)"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.next_id = 0
        logger.warning("🗑️  Database cleared!")


    def is_empty(self) -> bool:
        """Check if database is empty"""
        return self.index.ntotal == 0    
    
    def get_size(self) -> int:
        """Get number of vectors in database"""
        return self.index.ntotal
    
    def insert_batch(self, vectors: List[np.ndarray], metadatas: List[Dict]) -> bool:
        """Insert batch of vectors"""
        try:
            if isinstance(vectors, list):
                vectors = np.array(vectors)
            ids = self.add_vectors_batch(vectors, metadatas)
            return len(ids) > 0
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return False
        
    def insert(self, vector: np.ndarray, metadata: Dict) -> int:
        """Insert single vector (alias for add_vector)"""
        return self.add_vector(vector, metadata)    
    

    def has_hash(self, file_hash: str) -> bool:
        """Check if hash exists in metadata"""
        for meta in self.metadata:
            if meta.get('hash') == file_hash and not meta.get('deleted', False):
                return True
        return False




    # Utility functions for quick access
def create_database(dimension: int = 512) -> VectorDatabase:
    """Quick utility to create database"""
    return VectorDatabase(dimension=dimension)


def load_database(
    index_path: str = "data/vector_store/faiss_index.bin",
    metadata_path: str = "data/vector_store/metadata.pkl"
) -> VectorDatabase:
    """Quick utility to load existing database"""
    db = VectorDatabase(index_path=index_path, metadata_path=metadata_path)
    db.load()
    return db










                
        
    


        






        