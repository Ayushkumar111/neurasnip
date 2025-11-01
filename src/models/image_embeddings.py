"""CLIP Embedding generator
This will handle image and text embedding generation using open ai clip models
"""

import torch 
import clip
import numpy as np
from PIL import Image
from typing import List, Optional, Union , Tuple
from loguru import logger
import os 

# configuring our logger

logger.add("logs/image_embeddings.log", rotation="10 MB")

class CLIPEmbeddings:
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        image_weight: float = 0.7,
        text_weight: float = 0.3
    ):
        

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")



        self.model , self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.image_weight = image_weight
        self.text_weight = text_weight

        self.embedding_dim= self.model.visual.output_dim

        logger.info(f"Loaded CLIP model: {model_name} with embedding dimension: {self.embedding_dim}")
        logger.info(f"Image weight: {self.image_weight}, Text weight: {self.text_weight}")
         
    def generate_image_embedding(self , image:Image.Image)-> np.ndarray:

        """Generate image embedding using CLIP model

        Args:
            image (PIL.Image.Image): Input image

        Returns:
            np.ndarray: Image embedding
        """  
        try:
            
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            
            
            with torch.no_grad(): 
                image_features = self.model.encode_image(image_tensor)
                
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
           
            embedding = image_features.cpu().numpy().squeeze()
            
            logger.debug(f"Generated image embedding: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return np.zeros(self.embedding_dim)  

    def generate_text_embedding(self, text:str)-> np.ndarray:
        """ genearte emebedding vector for text
        arg - text: Input text string (e.g., OCR extracted text or query)
        return - numpy array of shape (embedding_dim,)
        """

        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding generation.")
                return np.zeros(self.embedding_dim)
            
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)

            # generate embedding

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)

                #normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                embedding = text_features.cpu().numpy().squeeze()
                logger.debug(f"Generated text embedding: shape {embedding.shape} | text length: {len(text)}")
                return embedding

        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return np.zeros(self.embedding_dim)


    def generate_hybrid_embedding(
            self,
            image: Image.Image,
            text:str
    )-> np.ndarray:
        """ genearte hybrid embedding from image and text
           arg - image: Input PIL Image
                 text: Input text string (e.g., OCR extracted text or query)
            return - numpy array of shape (embedding_dim,)
        
        """
        try:
            image_emb=self.generate_image_embedding(image)
            text_emb=self.generate_text_embedding(text)

            hybrid_emb = (self.image_weight * image_emb) + (self.text_weight * text_emb)

            norm = np.linalg.norm(hybrid_emb)
            if norm > 0:
                hybrid_emb = hybrid_emb / norm
            logger.debug(f"Generated hybrid embedding: shape {hybrid_emb.shape}")
            return hybrid_emb    
        except Exception as e:
            logger.error(f"Failed to generate hybrid embedding: {e}")
            return np.zeros(self.embedding_dim)

    def generate_batch_embeddings(
        self, 
        images: List[Image.Image]
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images at once (faster!)
        
        Args:
            images: List of PIL Images
            
        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        try:
            # Preprocess all images
            image_tensors = torch.stack([
                self.preprocess(img) for img in images
            ]).to(self.device)
            
            # Batch processing (more efficient than one-by-one)
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings = image_features.cpu().numpy()
            
            logger.info(f"Generated {len(images)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return np.zeros((len(images), self.embedding_dim))
        
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # cosine similarity = dot product (since vectors are normalized)
        # reason: Measures angle between vectors, perfect for semantic similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    
    def get_model_info(self) -> dict:
        """Get information about loaded model"""
        return {
            'model_name': 'CLIP',
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'image_weight': self.image_weight,
            'text_weight': self.text_weight
        }
    
    def generate_embedding(
    image: Image.Image, 
    text: Optional[str] = None,
    model_name: str = "ViT-B/32"
) -> np.ndarray:
        
        """ quick utility function to generate embedding
        arg - image: Input PIL Image
              text: Optional text string
              model_name: CLIP model name
        return - embedding vector      
        """
        embedder = CLIPEmbeddings(model_name=model_name)

        if text:
          return embedder.generate_hybrid_embedding(image, text)
    
        else:
         return embedder.generate_image_embedding(image)






