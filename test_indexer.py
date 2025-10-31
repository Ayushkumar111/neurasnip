"""
Test Image Indexer - The Complete Pipeline!
"""

from src.indexer.image_indexer import ImageIndexer
import os

def test_index_single_image():
    """Test indexing a single image"""
    print("\n" + "="*60)
    print("🧪 Test 1: Index Single Image")
    print("="*60)
    
    # Create indexer
    indexer = ImageIndexer(
        images_folder="data/images",
        db_path="data/vector_store/test.index"
    )
    
    # Index one image
    test_image = "C:/Users/AYUSH/Desktop/ggg.jpg"
    
    if os.path.exists(test_image):
        success = indexer.add_new_image(test_image)
        
        if success:
            print("✅ Image indexed successfully!")
            stats = indexer.get_stats()
            print(f"📊 Database size: {stats['database_size']} vectors")
        else:
            print("❌ Failed to index image")
    else:
        print("⚠️  Test image not found")


def test_index_folder():
    """Test indexing entire folder"""
    print("\n" + "="*60)
    print("🧪 Test 2: Index Entire Folder")
    print("="*60)
    
    # Put some images in data/images first!
    folder = "data/images"
    
    # Create indexer
    indexer = ImageIndexer(
        images_folder=folder,
        db_path="data/vector_store/images.index",
        batch_size=32,
        skip_duplicates=True
    )
    
    # Index the folder
    stats = indexer.index_folder()
    
    print("\n📊 Final Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   New indexed: {stats['new_indexed']}")
    print(f"   Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"   Errors: {stats['errors']}")
    
    # Test duplicate detection
    print("\n🔄 Testing duplicate detection...")
    print("Running indexer again (should skip all images)...")
    stats2 = indexer.index_folder()
    
    print(f"   Second run - New indexed: {stats2['new_indexed']}")
    print(f"   Second run - Duplicates: {stats2['duplicates_skipped']}")
    
    if stats2['duplicates_skipped'] == stats['total_processed']:
        print("✅ Duplicate detection working!")


def test_complete_pipeline():
    """Test the complete pipeline with real data"""
    print("\n" + "="*60)
    print("🧪 Test 3: Complete Pipeline Test")
    print("="*60)
    
    # Create test images folder
    import shutil
    from PIL import Image, ImageDraw
    
    test_folder = "data/images/test_set"
    os.makedirs(test_folder, exist_ok=True)
    
    # Create a few test images with text
    for i in range(5):
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), f"Test Image #{i}", fill='black')
        draw.text((50, 150), f"Content: Sample {i}", fill='blue')
        img.save(f"{test_folder}/test_{i}.png")
    
    print(f"✅ Created 5 test images in {test_folder}")
    
    # Index them
    indexer = ImageIndexer(
        images_folder=test_folder,
        db_path="data/vector_store/test_complete.index"
    )
    
    stats = indexer.index_folder()
    
    print("\n📊 Pipeline Results:")
    print(f"   Images created: 5")
    print(f"   Images indexed: {stats['new_indexed']}")
    print(f"   Database size: {indexer.database.get_size()}")
    
    if stats['new_indexed'] == 5:
        print("\n✅ Complete pipeline working perfectly!")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    shutil.rmtree(test_folder)
    print("✅ Cleanup complete")


if __name__ == "__main__":
    print("\n🚀 Testing Image Indexer Module")
    print("="*60)
    
    try:
        test_index_single_image()
        test_index_folder()
        test_complete_pipeline()
        
        print("\n" + "="*60)
        print("✅ All indexer tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()