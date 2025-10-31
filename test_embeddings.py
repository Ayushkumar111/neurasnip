"""
Test CLIP embeddings module
"""

from src.models.image_embeddings import CLIPEmbeddings
from src.utils.image_processor import ImageProcessor
from PIL import Image, ImageDraw
import numpy as np

def create_test_images():
    """Create test images with different content"""
    images = {}
    
    # Image 1: Cat-related
    img1 = Image.new('RGB', (400, 300), color='white')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((50, 100), "CAT", fill='black')
    draw1.text((50, 150), "Fluffy kitten", fill='blue')
    images['cat'] = img1
    
    # Image 2: Dog-related  
    img2 = Image.new('RGB', (400, 300), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((50, 100), "DOG", fill='black')
    draw2.text((50, 150), "Golden retriever", fill='green')
    images['dog'] = img2
    
    # Image 3: Car-related
    img3 = Image.new('RGB', (400, 300), color='white')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((50, 100), "CAR", fill='black')
    draw3.text((50, 150), "Tesla Model 3", fill='red')
    images['car'] = img3
    
    return images


def test_basic_embedding():
    """Test basic embedding generation"""
    print("\n" + "="*60)
    print("🧪 Test 1: Basic Image Embedding")
    print("="*60)
    
    embedder = CLIPEmbeddings(model_name="ViT-B/32")
    
    # Create simple test image
    test_img = Image.new('RGB', (224, 224), color='blue')
    
    # Generate embedding
    embedding = embedder.generate_image_embedding(test_img)
    
    print(f"✅ Embedding generated!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   Norm (should be ~1.0): {np.linalg.norm(embedding):.4f}")


def test_text_embedding():
    """Test text embedding"""
    print("\n" + "="*60)
    print("🧪 Test 2: Text Embedding")
    print("="*60)
    
    embedder = CLIPEmbeddings()
    
    texts = [
        "A cute cat sitting on a couch",
        "A happy dog playing in the park",
        "A red sports car on the highway"
    ]
    
    for text in texts:
        emb = embedder.generate_text_embedding(text)
        print(f"✅ '{text}'")
        print(f"   Embedding shape: {emb.shape}")


def test_similarity():
    """Test similarity between embeddings"""
    print("\n" + "="*60)
    print("🧪 Test 3: Embedding Similarity")
    print("="*60)
    
    embedder = CLIPEmbeddings()
    images = create_test_images()
    
    # Generate embeddings
    embeddings = {}
    for name, img in images.items():
        embeddings[name] = embedder.generate_image_embedding(img)
    
    # Compare similarities
    print("\n📊 Similarity Matrix:")
    print("-" * 60)
    
    for name1 in embeddings:
        for name2 in embeddings:
            sim = embedder.compute_similarity(
                embeddings[name1], 
                embeddings[name2]
            )
            print(f"{name1} ↔ {name2}: {sim:.4f}")
    
    print("\n💡 Interpretation:")
    print("   1.0000 = Identical")
    print("   0.9-1.0 = Very similar")
    print("   0.7-0.9 = Somewhat similar")
    print("   <0.7 = Different")


def test_hybrid_embedding():
    """Test hybrid image + text embedding"""
    print("\n" + "="*60)
    print("🧪 Test 4: Hybrid Embeddings")
    print("="*60)
    
    embedder = CLIPEmbeddings(image_weight=0.7, text_weight=0.3)
    
    # Create image with text
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), "RECEIPT", fill='black')
    draw.text((50, 150), "Coffee Shop - $15.99", fill='blue')
    
    text = "RECEIPT Coffee Shop - $15.99"
    
    # Generate all three types
    img_emb = embedder.generate_image_embedding(img)
    txt_emb = embedder.generate_text_embedding(text)
    hybrid_emb = embedder.generate_hybrid_embedding(img, text)
    
    print(f"✅ Image embedding shape: {img_emb.shape}")
    print(f"✅ Text embedding shape: {txt_emb.shape}")
    print(f"✅ Hybrid embedding shape: {hybrid_emb.shape}")
    
    # Compare how hybrid relates to each
    sim_to_img = embedder.compute_similarity(hybrid_emb, img_emb)
    sim_to_txt = embedder.compute_similarity(hybrid_emb, txt_emb)
    
    print(f"\n📊 Hybrid similarity to image: {sim_to_img:.4f}")
    print(f"📊 Hybrid similarity to text: {sim_to_txt:.4f}")
    print(f"💡 Hybrid combines both! (weighted 70% image, 30% text)")


def test_with_real_image():
    """Test with actual processed image"""
    print("\n" + "="*60)
    print("🧪 Test 5: Real Image with OCR")
    print("="*60)
    
    import os
    
    # Check if test image exists
    test_path = "C:/Users/AYUSH/Desktop/test.jpg"
    if not os.path.exists(test_path):
        print("⚠️  Test image not found, skipping...")
        return
    
    # Process image
    processor = ImageProcessor()
    result = processor.process_image_full(test_path)
    
    if not result:
        print("❌ Image processing failed")
        return
    
    print(f"✅ Processed: {result['filename']}")
    print(f"📝 OCR Text: {result['ocr_text'][:100]}...")
    
    # Generate hybrid embedding
    embedder = CLIPEmbeddings()
    hybrid_emb = embedder.generate_hybrid_embedding(
        result['image'],
        result['ocr_text']
    )
    
    print(f"✅ Generated hybrid embedding: {hybrid_emb.shape}")
    print(f"📊 Embedding norm: {np.linalg.norm(hybrid_emb):.4f}")
    print("🎉 Ready for vector database storage!")


if __name__ == "__main__":
    print("\n🚀 Testing CLIP Embeddings Module")
    print("="*60)
    
    try:
        test_basic_embedding()
        test_text_embedding()
        test_similarity()
        test_hybrid_embedding()
        test_with_real_image()
        
        print("\n" + "="*60)
        print("✅ All tests passed! CLIP embeddings module working!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()