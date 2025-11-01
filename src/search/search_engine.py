""" query engine our main search engine , this will handle natural language queires
and will return ranked results 
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from loguru import logger
from PIL import Image

from src.models.image_embeddings import CLIPEmbeddings
from src.database.vector_db import VectorDatabase
from src.utils.image_processor import ImageProcessor
from src.utils.color_detector import ColorDetector

logger.add("data/logs/search_engine.log", rotation="10 MB")

class SearchEngine:

    """Main search engine that handles
       text queries
       image queries 
       hybrid queries ( text + image)
    """

    def __init__(
        self,
        db_path:str="data/vector_store/images.index",
        model_name: str ="ViT-B/32"
    ):
        """
        Initialize the query engine
        arg - db_path - path to vector database
        modelname - clip model name

        
        """

        logger.info("="*60)
        logger.info("Initializing Search Engine")
        logger.info("="*60)

        #loading embedder
        self.embedder = CLIPEmbeddings(model_name=model_name)
        logger.info(f"Embedder loaded : {model_name}")

        #loading database

        self.database = VectorDatabase(
            index_path=db_path,
            metadata_path=db_path.replace('.index','_metadata.pkl')

        )
        #try to load the database

        if self.database.load():
            logger.info(f" Database loaded: {self.database.get_size()} vectors")
        else:
            logger.warning("  No existing database found - starting fresh")

        #image processor for handling image loading and preprocessing

        self.processor = ImageProcessor()

        logger.info("="*60)
        logger.info("✅ Query Engine Ready!")
        logger.info("="*60)    
   
    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[dict]:
        """
        🔍 Search using natural language query
        
        Args:
            query: Natural language text (e.g., "show me receipts from coffee shops")
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of results with images and metadata
        """
        logger.info(f"🔍 Text Query: '{query}'")
        
        try:
            # Check if database is empty
            if self.database.is_empty():
                logger.warning("Database is empty! Index some images first.")
                return []
            
            # Generate query embedding
            logger.debug("Generating query embedding...")
            query_vector = self.embedder.generate_text_embedding(query)
            
            # Search database
            logger.debug(f"Searching for top {top_k} results...")
            results = self.database.search(query_vector, top_k=top_k)
            
            # Filter by minimum similarity
            filtered_results = [
                r for r in results 
                if r['similarity'] >= min_similarity
            ]
            
            logger.info(f"✅ Found {len(filtered_results)} results (filtered from {len(results)})")
            
            # Add ranking info
            for i, result in enumerate(filtered_results, 1):
                result['rank'] = i
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[dict]:
        """
        🖼️ Search using an image (find similar images)
        
        Args:
            image_path: Path to query image
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar images
        """
        logger.info(f"🖼️ Image Query: {image_path}")
        
        try:
            # Load and process query image
            image = self.processor.load_image(image_path)
            if image is None:
                logger.error(f"Failed to load query image: {image_path}")
                return []
            
            # Generate image embedding
            logger.debug("Generating image embedding...")
            query_vector = self.embedder.generate_image_embedding(image)
            
            # Search
            results = self.database.search(query_vector, top_k=top_k + 1)
            
            # Filter out the query image itself (by path)
            filtered_results = [
                r for r in results
                if r.get('path') != image_path and r['similarity'] >= min_similarity
            ][:top_k]
            
            logger.info(f"✅ Found {len(filtered_results)} similar images")
            
            # Add ranking
            for i, result in enumerate(filtered_results, 1):
                result['rank'] = i
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Image search failed: {e}")
            return []
    
    
    def search_hybrid(
        self,
        query_text: str,
        query_image: Optional[str] = None,
        top_k: int = 10,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[dict]:
        """
        🎯 Hybrid search (combine text + image query)
        
        Args:
            query_text: Text query
            query_image: Optional image path
            top_k: Number of results
            text_weight: Weight for text component (0-1)
            image_weight: Weight for image component (0-1)
            
        Returns:
            Combined search results
        """
        logger.info(f"🎯 Hybrid Query: text='{query_text}', image={query_image}")
        
        try:
            # Generate text embedding
            text_emb = self.embedder.generate_text_embedding(query_text)
            
            if query_image:
                # Load image
                image = self.processor.load_image(query_image)
                if image:
                    # Generate image embedding
                    img_emb = self.embedder.generate_image_embedding(image)
                    
                    # Combine embeddings
                    query_vector = (text_weight * text_emb) + (image_weight * img_emb)
                    
                    # Normalize
                    norm = np.linalg.norm(query_vector)
                    if norm > 0:
                        query_vector = query_vector / norm
                else:
                    logger.warning("Image load failed, using text-only query")
                    query_vector = text_emb
            else:
                query_vector = text_emb
            
            # Search
            results = self.database.search(query_vector, top_k=top_k)
            
            logger.info(f"✅ Found {len(results)} hybrid results")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    
    def search_by_ocr(
        self,
        ocr_text: str,
        top_k: int = 10
    ) -> List[dict]:
        """
        📝 Search by OCR text content
        
        Args:
            ocr_text: Text to search for in OCR content
            top_k: Number of results
            
        Returns:
            Images containing similar text
        """
        logger.info(f"📝 OCR Search: '{ocr_text[:50]}...'")
        
        # Use text embedding search
        return self.search_by_text(ocr_text, top_k=top_k)
    
    
    def get_random_samples(self, count: int = 5) -> List[dict]:
        """
        🎲 Get random samples from database (for UI exploration)
        
        Args:
            count: Number of random samples
            
        Returns:
            List of random images
        """
        try:
            if self.database.is_empty():
                return []
            
            total = self.database.get_size()
            count = min(count, total)
            
            # Generate random indices
            import random
            indices = random.sample(range(total), count)
            
            results = []
            for idx in indices:
                if idx < len(self.database.metadata):
                    meta = self.database.metadata[idx].copy()
                    meta['rank'] = len(results) + 1
                    results.append(meta)
            
            logger.info(f"🎲 Retrieved {len(results)} random samples")
            return results
            
        except Exception as e:
            logger.error(f"Random sampling failed: {e}")
            return []
    
    
    def get_statistics(self) -> dict:
        """
        📊 Get search engine statistics
        
        Returns:
            Statistics dictionary
        """
        db_stats = self.database.get_stats()
        
        return {
            'total_images': db_stats['total_vectors'],
            'active_images': db_stats['active_vectors'],
            'embedding_dimension': db_stats['dimension'],
            'model_info': self.embedder.get_model_info(),
            'database_size': self.database.get_size()
        }
    
    
    def format_results(
        self,
        results: List[dict],
        include_thumbnails: bool = False
    ) -> List[dict]:
        """
        📋 Format results for display (clean up metadata)
        
        Args:
            results: Raw search results
            include_thumbnails: Load thumbnail images
            
        Returns:
            Formatted results
        """
        formatted = []
        
        for result in results:
            formatted_result = {
                'rank': result.get('rank', 0),
                'filename': result.get('filename', 'Unknown'),
                'path': result.get('path', ''),
                'similarity': result.get('similarity', 0.0),
                'similarity_percent': f"{result.get('similarity', 0.0) * 100:.1f}%",
                'ocr_preview': result.get('ocr_text', '')[:200],  # First 200 chars
                'has_text': bool(result.get('ocr_text', '').strip())
            }
            
            if include_thumbnails:
                try:
                    # Load thumbnail on demand
                    img = self.processor.load_image(result['path'])
                    if img:
                        formatted_result['thumbnail'] = self.processor.create_thumbnail(img)
                except Exception as e:
                    logger.error(f"Failed to load thumbnail: {e}")
            
            formatted.append(formatted_result)
        
        return formatted
    

    def search_by_text_with_color(
        self,
        query: str,
        color_filter: str = None,
        top_k: int = 10,
        min_similarity: float = 0.3
    ):
        """
        Search with optional color filtering
        
        Args:
            query: Text query
            color_filter: Color to filter (e.g., 'blue')
            top_k: Number of results
            min_similarity: Minimum similarity
            
        Returns:
            Filtered results
        """
        logger.info(f"🔍 Query: '{query}', Color filter: {color_filter}")
        
        # Get initial results (more than needed)
        results = self.search_by_text(
            query=query,
            top_k=top_k * 3,  # Get 3x more for filtering
            min_similarity=min_similarity
        )
        
        if not color_filter:
            return results[:top_k]
        
        # Filter by color
        logger.info(f"🎨 Filtering for color: {color_filter}")
        filtered_results = []
        
        for result in results:
            if ColorDetector.has_color(result['path'], color_filter):
                filtered_results.append(result)
                
                if len(filtered_results) >= top_k:
                    break
        
        logger.info(f"✅ Found {len(filtered_results)} results with {color_filter}")
        
        return filtered_results


def quick_search(
    query: str,
    db_path: str = "data/vector_store/images.index",
    top_k: int = 5
) -> List[dict]:
    """
    🚀 Quick utility to search with one line
    
    Example:
        results = quick_search("show me receipts")
    """
    engine = SearchEngine(db_path=db_path)
    return engine.search_by_text(query, top_k=top_k)



def search_and_print(query: str, top_k: int = 5):
    """
    🖨️ Search and print results (for testing)
    
    Example:
        search_and_print("coffee shop receipts")
    """
    results = quick_search(query, top_k=top_k)
    
    print(f"\n🔍 Query: '{query}'")
    print("="*60)
    
    if not results:
        print("❌ No results found")
        return
    
    for r in results:
        print(f"\n{r['rank']}. {r['filename']}")
        print(f"   📊 Similarity: {r['similarity']:.4f}")
        print(f"   📍 Path: {r['path']}")
        if r.get('ocr_text'):
            preview = r['ocr_text'][:100]
            print(f"   📝 OCR: {preview}...")
        print("-"*60)
    
    print(f"\n✅ Found {len(results)} results")


