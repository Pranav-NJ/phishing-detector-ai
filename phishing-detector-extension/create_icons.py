from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    # Create a new image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a shield shape
    draw.ellipse((0, 0, size-1, size-1), fill='#4285F4', outline='#3367D6')
    
    # Add a checkmark (✓) in the center
    try:
        font_size = size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = "✓"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    position = ((size - text_width) // 2, (size - text_height) // 2 - size//10)
    draw.text(position, text, fill='white', font=font)
    
    # Save the image
    img.save(output_path, 'PNG')

# Create icons directory if it doesn't exist
os.makedirs('icons', exist_ok=True)

# Create different icon sizes
sizes = [16, 48, 128]
for size in sizes:
    create_icon(size, f'icons/icon{size}.png')

print("Icons created successfully in the 'icons' directory!")
