""" image indexer module , this module will coordinate all modules to index images into database"""

import os 
from pathlib import Path
from typing import List , Dict , Optional
from loguru import logger
from tqdm import tqdm 

from src.utils.image_processor import ImageProcessor
from src.models.image_embeddings import CLIPEmbeddings
from src.database.vector_db import VectorDatabase

logger.add("data/logs/image_indexer.log", rotation="10 MB")    

class ImageIndexer:
    """
    Indexes images by processing them and storing in vector db coordinate image processor , clipembeding and vectordb
    """

    def __init__(
        self,
        images_folder: str = "D:/IMAGES",
        db_path: str = "data/vector_store/images.index",
        batch_size: int = 32,
        skip_duplicates: bool = True,
    ):
        
        self.images_folder = images_folder
        
        self.batch_size = batch_size
        self.skip_duplicates = skip_duplicates

        logger.info("Initializing ImageIndexer...")

        self.processor = ImageProcessor()
        logger.info("ImageProcessor initialized.")
        self.embedder = CLIPEmbeddings()
        logger.info("CLIPEmbeddings initialized.")
        self.database = VectorDatabase(
             index_path=db_path,
             metadata_path=db_path.replace('.index', '_metadata.pkl')
        )
        logger.info("VectorDatabase initialized.")

        self.stats = {
            'total_processed': 0,
            'new_indexed': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
        
        logger.info(f"🚀 Indexer initialized | Batch size: {batch_size}")


    def get_image_files(self)-> List[str]:
        """ scan folders and get all the image files
        return - list of image file paths
        """   
        image_extensions = self.processor.supported_formats

        image_files = []
        for ext in image_extensions:
            pattern = f"**/*{ext}" 
            image_files.extend(Path(self.images_folder).glob(pattern))

        image_files = [str(f) for f in image_files]    
        logger.info(f"Found {len(image_files)} image files in {self.images_folder}")
        return image_files
    
    def is_already_indexed(self, image_hash: str)-> bool:
        """ check if image is already in database ( by hash)
         arg - image_hash: md5 hash of image file
         return - true if already indexed , false otherwise
        
        
        """

        if not self.skip_duplicates:
            return False
        
        return self.database.has_hash(image_hash)
    
    def index_single_image(self, image_path: str) -> bool:
        """
        Process and index a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if successfully indexed, False otherwise
        """
        try:
            # process image (Module 1)
            # why: Extract OCR text, preprocess image
            result = self.processor.process_image_full(image_path)
            
            if result is None:
                logger.error(f"Failed to process: {image_path}")
                self.stats['errors'] += 1
                return False
            
            # check if already indexed
            if self.is_already_indexed(result['hash']):
                logger.debug(f"⏭  Skipping duplicate: {result['filename']}")
                self.stats['duplicates_skipped'] += 1
                return False
            
            # generate hybrid embedding 
            # why: combine visual features + OCR text
            embedding = self.embedder.generate_hybrid_embedding(
                result['image'],
                result['ocr_text']
            )
            
            # store in database
            
            metadata = {
                'filename': result['filename'],
                'path': result['path'],
                'ocr_text': result['ocr_text'],
                'hash': result['hash']
            }
            
            self.database.insert(embedding, metadata)
            
            # update statistics
            self.stats['new_indexed'] += 1
            self.stats['total_processed'] += 1
            
            logger.info(f" Indexed: {result['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing {image_path}: {e}")
            self.stats['errors'] += 1
            return False
        



    def index_batch(self, image_paths: List[str]) -> int:
        """
        Process and index multiple images efficiently
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Number of successfully indexed images
        """
        # batch processing:much faster for large folders
        
        indexed_count = 0
        
        # results list
        results = []
        
        
        logger.info(f"Processing batch of {len(image_paths)} images...")
        
        for img_path in image_paths:
            result = self.processor.process_image_full(img_path)
            if result:
                # check duplicates
                if self.skip_duplicates and self.is_already_indexed(result['hash']):
                    self.stats['duplicates_skipped'] += 1
                    logger.debug(f"⏭  Skipping: {result['filename']}")
                    continue
                
                results.append(result)
            else:
                self.stats['errors'] += 1
        
        if not results:
            logger.warning("No valid images in batch")
            return 0
        
        
        logger.info(f"Generating embeddings for {len(results)} images...")
        
        images = [r['image'] for r in results]
        texts = [r['ocr_text'] for r in results]
        
        
        embeddings = []
        for img, txt in zip(images, texts):
            emb = self.embedder.generate_hybrid_embedding(img, txt)
            embeddings.append(emb)
        
       
        logger.info(f"Inserting {len(embeddings)} vectors into database...")
        
        metadatas = [
            {
                'filename': r['filename'],
                'path': r['path'],
                'ocr_text': r['ocr_text'],
                'hash': r['hash']
            }
            for r in results
        ]
        
       
        success = self.database.insert_batch(embeddings, metadatas)
        
        if success:
            indexed_count = len(embeddings)
            self.stats['new_indexed'] += indexed_count
            self.stats['total_processed'] += len(image_paths)
        
        return indexed_count    
    


    def index_folder(self, use_batch: bool = True) -> Dict:
        """
        Index all images in the configured folder
        
        Args:
            use_batch: Use batch processing faster
            
        Returns:
            Statistics dictionary
        """
        logger.info("="*60)
        logger.info(" Starting folder indexing")
        logger.info("="*60)
        
        # reset statistics
        self.stats = {
            'total_processed': 0,
            'new_indexed': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
        
        # get all image files
        image_files = self.get_image_files()
        
        if not image_files:
            logger.warning(f"No images found in {self.images_folder}")
            return self.stats
        
        logger.info(f"📊 Found {len(image_files)} images to process")
        
        # process images
        if use_batch and len(image_files) > self.batch_size:
            
            logger.info(f"Using batch processing (batch_size={self.batch_size})")
            
            # split into batches
            # process 32 images at a time for efficiency
            for i in tqdm(range(0, len(image_files), self.batch_size), desc="Indexing batches"):
                batch = image_files[i:i + self.batch_size]
                self.index_batch(batch)
        
        else:
            # single image processing
            logger.info("Processing images one by one...")
            
            for img_path in tqdm(image_files, desc="Indexing images"):
                self.index_single_image(img_path)
        
       
        logger.info("💾 Saving database...")
        self.database.save()
        
        # Print summary
        logger.info("="*60)
        logger.info(" Indexing complete!")
        logger.info(f" Statistics:")
        logger.info(f"   Total processed: {self.stats['total_processed']}")
        logger.info(f"   New indexed: {self.stats['new_indexed']}")
        logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"   Errors: {self.stats['errors']}")
        logger.info(f"   Database size: {self.database.get_size()} vectors")
        logger.info("="*60)
        
        return self.stats



        
    def add_new_image(self, image_path: str) -> bool:
        """
        Add a single new image to the index
        Useful for adding images one at a time
        
        Args:
            image_path: Path to new image
            
        Returns:
            True if successfully added
        """
        logger.info(f"Adding new image: {image_path}")
        
        success = self.index_single_image(image_path)
        
        if success:
            
            self.database.save()
            logger.info(f"Image added and database saved")
        
        return success
    


    def rebuild_index(self, force: bool = False):
        """
        Rebuild entire index from scratch
        
        Args:
            force: Force rebuild even if database exists
        """
        if force or not self.database.is_empty():
            logger.warning("⚠️  Clearing existing database...")
            self.database.clear()
        
        logger.info("🔄 Rebuilding index from scratch...")
        self.index_folder()
    
    
    def get_stats(self) -> Dict:
        """Get current indexing statistics"""
        return {
            **self.stats,
            'database_size': self.database.get_size(),
            'images_folder': self.images_folder
        }
    

    # Utility function for quick access
def index_images(
    images_folder: str,
    db_path: str = "data/vector_store/images.index"
) -> Dict:
    """
    Quick utility to index a folder
    
    Args:
        images_folder: Path to images
        db_path: Path to save database
        
    Returns:
        Statistics dictionary
    """
    indexer = ImageIndexer(images_folder, db_path)
    return indexer.index_folder()



if __name__ == "__main__":
    """
    Run indexer from command line
    Usage: python -m src.indexer.image_indexer
    """
    import sys
    
    logger.info("="*60)
    logger.info("🚀 Starting Image Indexing")
    logger.info("="*60)
    
    try:
        # Create indexer with default paths
        indexer = ImageIndexer()
        
        # Index all images
        result = indexer.index_folder()
        
        # Print summary
        logger.info("="*60)
        logger.info("✅ Indexing Complete!")
        logger.info("="*60)
        logger.success(f"📊 New images indexed: {result['new_indexed']}")
        logger.success(f"⏭️  Skipped (duplicates): {result['duplicates_skipped']}")
        logger.success(f"❌ Failed: {result['errors']}")
        
        logger.info("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if result['errors'] == 0 else 1)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Indexing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)