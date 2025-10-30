import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageOps
import pytesseract
from loguru import logger


# configure logger

logger.add("data/logs/image_processor.log", rotation="10 MB")


class ImageProcessor:
    """ Handle all image processing operations"""

    def __init__(self,max_image_size: Tuple[int,int]=(800,800)):
        
        self.max_image_size= max_image_size
        self.supported_formats ={'.jpg' , '.jpeg' , '.png' , '.bmp' , '.tiff'  , '.gif' , '.webp' }
        logger.info(f"ImageProcessor initialized with max size : {max_image_size}")

    def is_supported_image(self , file_path: str)-> bool:

        return Path(file_path).suffix.lower() in self.supported_formats

    def load_image(self, image_path: str)-> Optional[Image.Image]:

        try:
            if not  os.path.exists(image_path):
                logger.error(f"Image not found at path: {image_path}")
                return None
            
            if not self.is_supported_image(image_path):
                logger.warning(f"Unsupported image format: {image_path}")
                return None 
            
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            logger.debug(f"Image loaded successfully: {image_path} with size {img.size}")
            return img  
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
        

    def preprocess_image(self, image: Image.Image)-> Image.Image:

        try:
            if image.size[0]> self.max_image_size[0] or image.size[1]> self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                logger.debug(f"Image resized to {image.size}")
            return image
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
        
    def extract_text_ocr(self, image: Image.Image)-> str:


        try:

            text = pytesseract.image_to_string(image,lang='eng')
            logger.debug(f"OCR extracted text of length {len(text)}")

            text = text.strip()

            if text:
                logger.debug(f"OCR extracter {len(text)} characters")
            else:
                logger.debug("No text found in image via OCR")

            return text        
        
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return ""
        
    def generate_image_hash(self,image_path:str)-> str:

        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            logger.debug(f"Generated hash for {image_path}: {file_hash}")
            return file_hash

        except Exception as e:
            logger.error(f"Error generating hash for {image_path}: {e}")
            return ""    
        
    def create_thumbnail(self,image: Image.Image, size: Tuple[int,int]=(200,200))-> Image.Image:

        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            logger.debug(f"Thumbnail created with size {thumbnail.size}")
            return thumbnail
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return image
        
    def process_image_full(self,image_path:str)-> Optional[dict]:
      

      try:
        image = self.load_image(image_path)
        if image is None:
            return None
        
        preprocessed_image = self.preprocess_image(image)
        ocr_text = self.extract_text_ocr(preprocessed_image)
        image_hash = self.generate_image_hash(image_path)
        thumbnail = self.create_thumbnail(preprocessed_image)

        result = {
            'image': preprocessed_image,           
            'ocr_text': ocr_text,
            'hash': image_hash,                    
            'thumbnail': thumbnail,
            'filename': os.path.basename(image_path),  
            'path': os.path.abspath(image_path)    
        }

        logger.info(f"Full image processing completed for {image_path}")
        return result
      except Exception as e:
        logger.error(f"Error in full image processing for {image_path}: {e}")
        return None
      

    def process_image(image_path:str) -> Optional[dict]:
        processor = ImageProcessor()
        return processor.process_image_full(image_path)
    
    