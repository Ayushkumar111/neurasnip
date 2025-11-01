"""
Color detection utility for filtering search results
"""

import numpy as np
from PIL import Image
from collections import Counter
import colorsys


class ColorDetector:
    """Detect dominant colors in images"""
    
    # Color definitions (HSV ranges)
    COLOR_RANGES = {
        'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        'blue': [(100, 50, 50), (130, 255, 255)],
        'green': [(40, 50, 50), (80, 255, 255)],
        'yellow': [(20, 50, 50), (40, 255, 255)],
        'orange': [(10, 50, 50), (20, 255, 255)],
        'purple': [(130, 50, 50), (170, 255, 255)],
        'pink': [(140, 50, 50), (170, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
        'white': [(0, 0, 200), (180, 50, 255)],
        'gray': [(0, 0, 50), (180, 50, 200)]
    }
    
    @staticmethod
    def get_dominant_colors(image_path: str, top_n: int = 3) -> list:
        """
        Get dominant colors from image
        
        Args:
            image_path: Path to image
            top_n: Number of dominant colors to return
            
        Returns:
            List of color names
        """
        try:
            # Load image
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize for faster processing
            img = img.resize((100, 100))
            
            # Convert to HSV
            pixels = np.array(img)
            hsv_pixels = []
            
            for row in pixels:
                for pixel in row:
                    r, g, b = pixel / 255.0
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_pixels.append((int(h * 180), int(s * 255), int(v * 255)))
            
            # Detect colors
            color_counts = Counter()
            
            for h, s, v in hsv_pixels:
                for color_name, ranges in ColorDetector.COLOR_RANGES.items():
                    if len(ranges) == 2:  # Single range
                        low, high = ranges
                        if (low[0] <= h <= high[0] and 
                            low[1] <= s <= high[1] and 
                            low[2] <= v <= high[2]):
                            color_counts[color_name] += 1
                            break
                    else:  # Multiple ranges (for red wrapping around)
                        low1, high1, low2, high2 = ranges
                        if ((low1[0] <= h <= high1[0] or low2[0] <= h <= high2[0]) and 
                            low1[1] <= s <= high1[1] and 
                            low1[2] <= v <= high1[2]):
                            color_counts[color_name] += 1
                            break
            
            # Return top N colors
            return [color for color, _ in color_counts.most_common(top_n)]
            
        except Exception as e:
            print(f"Color detection failed: {e}")
            return []
    
    @staticmethod
    def has_color(image_path: str, target_color: str, threshold: float = 0.1) -> bool:
        """
        Check if image has specific color
        
        Args:
            image_path: Path to image
            target_color: Color to look for (e.g., 'blue')
            threshold: Minimum percentage of color (0-1)
            
        Returns:
            True if color is present above threshold
        """
        dominant_colors = ColorDetector.get_dominant_colors(image_path, top_n=5)
        return target_color.lower() in [c.lower() for c in dominant_colors]