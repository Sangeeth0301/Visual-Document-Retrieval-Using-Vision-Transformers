import os
from PIL import Image, ImageDraw, ImageFont

def create_dumb_pdf(text_lines, filename, size=(800, 1000), bg_color="white"):
    """
    Creates a simple image with text and saves it as a PDF.
    This simulates a document page for our educational baseline.
    """
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a reasonable font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        title_font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
        title_font = font
        
    y_text = 100
    for i, line in enumerate(text_lines):
        f = title_font if i == 0 else font
        draw.text((100, y_text), line, fill="black", font=f)
        y_text += 80

    output_dir = "data/pdfs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    img.save(output_path, "PDF", resolution=100.0)
    print(f"Generated: {output_path}")

def generate_samples():
    docs = [
        ("chennai_tourism.pdf", [
            "Chennai Tourism Guide",
            "Visit Marina Beach, the longest natural urban beach.",
            "Explore the ancient Kapaleeshwarar Temple in Mylapore.",
            "Enjoy authentic South Indian filter coffee."
        ]),
        ("pongal_festival.pdf", [
            "Tamil Nadu Festival Calendar",
            "Pongal is a four-day harvest festival celebrated in January.",
            "It marks the beginning of the sun's six-month-long journey.",
            "People prepare sweet Pongal dish to offer to the Sun God."
        ]),
        ("temple_architecture.pdf", [
            "Dravidian Architecture and Maps",
            "Temples are known for their towering gopurams (gateways).",
            "The Brihadisvara Temple at Thanjavur is a prime example.",
            "Intricate carvings depict mythological stories."
        ])
    ]
    
    for filename, lines in docs:
        create_dumb_pdf(lines, filename)

if __name__ == "__main__":
    generate_samples()
