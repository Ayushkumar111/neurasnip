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
        progress_callback = None,  # streamlit progress callback
    ):
        
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.skip_duplicates = skip_duplicates
        self.progress_callback = progress_callback  # call back to track progress in ui 

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
        
        logger.info(f" Indexer initialized | Batch size: {batch_size}")


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
            # process image we will use our module  1 here 
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
            
            #hybrid embeddings gg
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
            
            # update statistics stats are very much needed atp
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
        Process and index multiple images efficiently in a batch 
        
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
                # check duplicates really important man using hash to avoid reindexing
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
        
        total_images = len(image_files)
        logger.info(f"📊 Found {total_images} images to process")
        
        
        if self.progress_callback:
            self.progress_callback(0, total_images, "Starting indexing...")
        
        # process images
        if use_batch and len(image_files) > self.batch_size:
            logger.info(f"Using batch processing (batch_size={self.batch_size})")
            
            # split into batches goated 
            batches_total = (len(image_files) + self.batch_size - 1) // self.batch_size
            
            for batch_idx, i in enumerate(range(0, len(image_files), self.batch_size)):
                batch = image_files[i:i + self.batch_size]
                self.index_batch(batch)
                
                # report after processing each batch
                processed = min(i + self.batch_size, total_images)
                if self.progress_callback:
                    self.progress_callback(
                        processed, 
                        total_images,
                        f"Processing batch {batch_idx + 1}/{batches_total}"
                    )
        
        else:
            # single image processing
            logger.info("Processing images one by one...")
            
            for idx, img_path in enumerate(image_files):
                self.index_single_image(img_path)
                
                
                if self.progress_callback and (idx + 1) % 5 == 0: 
                    self.progress_callback(
                        idx + 1,
                        total_images,
                        f"Processed {idx + 1}/{total_images} images"
                    )
        
        # ✅report saving progress
        if self.progress_callback:
            self.progress_callback(total_images, total_images, "Saving database...")
        
        logger.info("💾 Saving database...")
        self.database.save()
        
        # this is final completion update 
        if self.progress_callback:
            self.progress_callback(
                total_images, 
                total_images, 
                f" Complete! Indexed {self.stats['new_indexed']} new images"
            )
        
        # summary to be printed 
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
        
        logger.info(" Rebuilding index from scratch...")
        self.index_folder()
    
    
    def get_stats(self) -> Dict:
        """Get current indexing statistics"""
        return {
            **self.stats,
            'database_size': self.database.get_size(),
            'images_folder': self.images_folder
        }
    

    #  this is just a utility function for quick access
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
    logger.info(" Starting Image Indexing")
    logger.info("="*60)
    
    try:
        # creating indexer instance
        indexer = ImageIndexer()
        
        # indexing all the images
        result = indexer.index_folder()
        
        #sumarry
        logger.info("="*60)
        logger.info("✅ Indexing Complete!")
        logger.info("="*60)
        logger.success(f" New images indexed: {result['new_indexed']}")
        logger.success(f"⏭  Skipped (duplicates): {result['duplicates_skipped']}")
        logger.success(f" Failed: {result['errors']}")
        
        logger.info("="*60)
        
        # exiting  with appropriate code
        sys.exit(0 if result['errors'] == 0 else 1)
        
    except KeyboardInterrupt:
        logger.warning("\n  Indexing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)