"""
Test Vector Database module
"""

from src.database.vector_db import VectorDatabase
from src.models.image_embeddings import CLIPEmbeddings
from src.utils.image_processor import ImageProcessor
import numpy as np
from PIL import Image, ImageDraw

def test_basic_operations():
    """Test basic add and search"""
    print("\n" + "="*60)
    print("🧪 Test 1: Basic Operations")
    print("="*60)
    
    # Create database
    db = VectorDatabase(dimension=512)
    
    # Create some dummy vectors
    vec1 = np.random.rand(512).astype('float32')
    vec2 = np.random.rand(512).astype('float32')
    vec3 = np.random.rand(512).astype('float32')
    
    # Add vectors with metadata
    id1 = db.add_vector(vec1, {'filename': 'cat.jpg', 'label': 'cat'})
    id2 = db.add_vector(vec2, {'filename': 'dog.jpg', 'label': 'dog'})
    id3 = db.add_vector(vec3, {'filename': 'car.jpg', 'label': 'car'})
    
    print(f"✅ Added 3 vectors")
    print(f"   IDs: {id1}, {id2}, {id3}")
    
    # Search
    results = db.search(vec1, top_k=2)
    print(f"\n🔍 Search for vec1 (cat):")
    for r in results:
        print(f"   {r['filename']}: similarity = {r['similarity']:.4f}")


def test_save_load():
    """Test persistence"""
    print("\n" + "="*60)
    print("🧪 Test 2: Save & Load")
    print("="*60)
    
    # Create and populate database
    db = VectorDatabase()
    vec = np.random.rand(512).astype('float32')
    db.add_vector(vec, {'filename': 'test.jpg'})
    
    # Save
    db.save()
    print("✅ Database saved")
    
    # Load in new instance
    db2 = VectorDatabase()
    db2.load()
    print(f"✅ Database loaded: {db2.index.ntotal} vectors")
    
    # Verify
    stats = db2.get_stats()
    print(f"📊 Stats: {stats}")


def test_with_real_embeddings():
    """Test with actual CLIP embeddings"""
    print("\n" + "="*60)
    print("🧪 Test 3: Real CLIP Embeddings")
    print("="*60)
    
    # Create test images
    images = {}
    labels = ['cat', 'dog', 'car']
    
    for label in labels:
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), label.upper(), fill='black')
        images[label] = img
    
    # Generate embeddings
    embedder = CLIPEmbeddings()
    db = VectorDatabase()
    
    print("📸 Adding images to database...")
    for label, img in images.items():
        vector = embedder.generate_image_embedding(img)
        db.add_vector(vector, {
            'filename': f'{label}.jpg',
            'label': label
        })
        print(f"   ✅ {label}.jpg")
    
    # Search with text query
    print("\n🔍 Searching with text queries...")
    
    queries = ['a cat', 'a dog', 'a car']
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = db.search_by_text(query, embedder, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"      {i}. {r['filename']} (similarity: {r['similarity']:.4f})")


def test_batch_operations():
    """Test batch add"""
    print("\n" + "="*60)
    print("🧪 Test 4: Batch Operations")
    print("="*60)
    
    db = VectorDatabase()
    
    # Create batch of vectors
    vectors = np.random.rand(10, 512).astype('float32')
    metadata = [{'filename': f'image_{i}.jpg'} for i in range(10)]
    
    # Add batch
    ids = db.add_vectors_batch(vectors, metadata)
    print(f"✅ Added batch: {len(ids)} vectors")
    
    # Search
    query = vectors[0]  # Search for first vector
    results = db.search(query, top_k=3)
    print(f"\n🔍 Top 3 results:")
    for r in results:
        print(f"   {r['filename']}: {r['similarity']:.4f}")


if __name__ == "__main__":
    print("\n🚀 Testing Vector Database Module")
    print("="*60)
    
    try:
        test_basic_operations()
        test_save_load()
        test_with_real_embeddings()
        test_batch_operations()
        
        print("\n" + "="*60)
        print("✅ All tests passed! Vector database working!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()