"""
Test Search Engine Module - The Final Boss Test! 🎮
"""

from src.search.search_engine import SearchEngine, quick_search
from src.indexer.image_indexer import ImageIndexer
from PIL import Image, ImageDraw
import os
import shutil


def setup_test_data():
    """Create test images with different content"""
    print("\n📸 Setting up test data...")
    print("="*60)
    
    test_folder = "data/images/search_test"
    os.makedirs(test_folder, exist_ok=True)
    
    # Create test images with specific content
    test_images = {
        'coffee_receipt.png': {
            'text': "STARBUCKS\nCoffee $4.50\nTotal: $4.50",
            'color': 'lightyellow'
        },
        'grocery_receipt.png': {
            'text': "WALMART\nMilk $3.99\nBread $2.49\nTotal: $6.48",
            'color': 'lightblue'
        },
        'cat_photo.png': {
            'text': "My cute cat\nFluffy kitty",
            'color': 'lightpink'
        },
        'car_photo.png': {
            'text': "Tesla Model 3\nRed sports car",
            'color': 'lightcoral'
        },
        'invoice.png': {
            'text': "INVOICE #12345\nWebsite Design\n$500.00",
            'color': 'lightgreen'
        }
    }
    
    for filename, data in test_images.items():
        img = Image.new('RGB', (400, 300), color=data['color'])
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), data['text'], fill='black')
        img.save(os.path.join(test_folder, filename))
    
    print(f"✅ Created {len(test_images)} test images")
    return test_folder


def test_indexing_first():
    """Index test images first"""
    print("\n" + "="*60)
    print("🧪 Test 1: Index Test Images")
    print("="*60)
    
    test_folder = setup_test_data()
    
    # Index the test folder
    indexer = ImageIndexer(
        images_folder=test_folder,
        db_path="data/vector_store/search_test.index",
        skip_duplicates=True
    )
    
    stats = indexer.index_folder()
    
    print(f"\n✅ Indexing complete!")
    print(f"   Indexed: {stats['new_indexed']} images")
    print(f"   Database size: {indexer.database.get_size()}")
    
    return "data/vector_store/search_test.index"


def test_text_search(db_path: str):
    """Test natural language search"""
    print("\n" + "="*60)
    print("🧪 Test 2: Natural Language Search")
    print("="*60)
    
    engine = SearchEngine(db_path=db_path)
    
    queries = [
        "show me receipts",
        "coffee shop",
        "cute cat",
        "car photo",
        "invoice"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-"*60)
        
        results = engine.search_by_text(query, top_k=3)
        
        if results:
            for r in results:
                print(f"   {r['rank']}. {r['filename']}")
                print(f"      Similarity: {r['similarity']:.4f}")
        else:
            print("   No results found")


def test_image_search(db_path: str):
    """Test search by similar image"""
    print("\n" + "="*60)
    print("🧪 Test 3: Image Similarity Search")
    print("="*60)
    
    engine = SearchEngine(db_path=db_path)
    
    query_image = "data/images/search_test/coffee_receipt.png"
    
    if os.path.exists(query_image):
        print(f"🖼️ Query Image: {query_image}")
        print("-"*60)
        
        results = engine.search_by_image(query_image, top_k=3)
        
        for r in results:
            print(f"   {r['rank']}. {r['filename']}")
            print(f"      Similarity: {r['similarity']:.4f}")
    else:
        print("⚠️ Query image not found")


def test_statistics(db_path: str):
    """Test statistics"""
    print("\n" + "="*60)
    print("🧪 Test 4: Engine Statistics")
    print("="*60)
    
    engine = SearchEngine(db_path=db_path)
    stats = engine.get_statistics()
    
    print("\n📊 Statistics:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Model: {stats['model_info']['model_name']}")


def test_quick_search():
    """Test quick utility function"""
    print("\n" + "="*60)
    print("🧪 Test 5: Quick Search Utility")
    print("="*60)
    
    print("\n🚀 Using quick_search()...")
    results = quick_search(
        "coffee",
        db_path="data/vector_store/search_test.index",
        top_k=3
    )
    
    print(f"✅ Found {len(results)} results using quick_search()")


def test_formatted_results(db_path: str):
    """Test result formatting"""
    print("\n" + "="*60)
    print("🧪 Test 6: Formatted Results")
    print("="*60)
    
    engine = SearchEngine(db_path=db_path)
    
    results = engine.search_by_text("receipt", top_k=3)
    formatted = engine.format_results(results)
    
    print("\n📋 Formatted results:")
    for r in formatted:
        print(f"\n   Rank: {r['rank']}")
        print(f"   File: {r['filename']}")
        print(f"   Match: {r['similarity_percent']}")
        print(f"   Has Text: {r['has_text']}")
        if r['ocr_preview']:
            print(f"   Preview: {r['ocr_preview'][:50]}...")


def cleanup_test_data():
    """Remove test data"""
    print("\n🧹 Cleaning up test data...")
    
    test_folder = "data/images/search_test"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
        print("✅ Test folder removed")
    
    # Remove test index files
    test_index = "data/vector_store/search_test.index"
    if os.path.exists(test_index):
        os.remove(test_index)
        os.remove(test_index.replace('.index', '_metadata.pkl'))
        print("✅ Test database removed")


if __name__ == "__main__":
    print("\n🚀 Testing Search Engine Module")
    print("="*60)
    
    try:
        # Run tests
        db_path = test_indexing_first()
        test_text_search(db_path)
        test_image_search(db_path)
        test_statistics(db_path)
        test_quick_search()
        test_formatted_results(db_path)
        
        print("\n" + "="*60)
        print("✅ All search tests passed!")
        print("="*60)
        
        # Cleanup
        cleanup_test_data()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()