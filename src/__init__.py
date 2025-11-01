"""
NeuraSnip - Semantic Image Search Engine
Main package initialization
"""

__version__ = '0.1.0'
__author__ = 'Ayush kumar'

# Make main classes easily importable
from .search import SearchEngine
from .indexer import ImageIndexer
from .database import VectorDatabase
from .models import CLIPEmbeddings
from .utils import ImageProcessor

__all__ = [
    'SearchEngine',
    'ImageIndexer',
    'VectorDatabase',
    'CLIPEmbeddings',
    'ImageProcessor'
]