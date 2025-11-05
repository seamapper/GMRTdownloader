#!/usr/bin/env python3
"""
Convert mgds.png to ico format for executable icon
"""

from PIL import Image
import os

def convert_png_to_ico():
    """Convert mac.png to mac.ico"""
    png_path = "media/GMRT-logo2020.png"
    ico_path = "media/GMRT-logo2020.ico"
    
    if not os.path.exists(png_path):
        print(f"Error: {png_path} not found!")
        return False
    
    try:
        # Open the PNG image
        img = Image.open(png_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create multiple sizes for the icon (Windows standard sizes)
        sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(ico_path, format='ICO', sizes=sizes)
        
        print(f"Successfully converted {png_path} to {ico_path}")
        return True
        
    except Exception as e:
        print(f"Error converting image: {e}")
        return False

if __name__ == "__main__":
    convert_png_to_ico() 