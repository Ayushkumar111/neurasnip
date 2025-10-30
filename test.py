

from src.utils.image_processor import ImageProcessor
from PIL import Image, ImageDraw
import os
from pathlib import Path




def test_single_image(image_path: str):
    """Test processing a single image"""
    print(f"🔍 Processing: {image_path}")
    print("-" * 60)
    
    # Initialize processor
    processor = ImageProcessor(max_image_size=(800, 800))
    
    # Process the image
    result = processor.process_image_full(image_path)
    
    if result:
        print("✅ Processing Successful!\n")
        print(f"📁 Filename:      {result.get('filename', 'N/A')}")
        print(f"📍 Full Path:     {result.get('path', 'N/A')}")
        print(f"📏 Image Size:   {result['image'].size[0]}x{result['image'].size[1]} pixels")
        print(f"🔑 File Hash:    {result['hash']}")
        print(f"\n📝 OCR Extracted Text:")
        print("─" * 60)
        if result['ocr_text']:
            print(result['ocr_text'])
            print("─" * 60)
            print(f"📊 Total Characters: {len(result['ocr_text'])}")
            print(f"📊 Total Words:      {len(result['ocr_text'].split())}")
        else:
            print("(No text found in image)")
            print("─" * 60)
        
        return result
    else:
        print("❌ Processing failed!")
        return None



if __name__ == "__main__":
    
    
  
    test_single_image("C:/Users/AYUSH/Desktop/test.jpg")
    
    
    print("\n🎉 Test complete!")




    