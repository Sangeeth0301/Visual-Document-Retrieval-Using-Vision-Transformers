import os
import json
from PIL import Image, ImageDraw, ImageFont

def create_dumb_image(text_lines, filename, size=(800, 1000), bg_color="white"):
    """
    Creates a simple image with text and saves it.
    This simulates a realistic document scan.
    """
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
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

    img.save(filename, resolution=100.0)

def generate_samples(base_dir="data"):
    # Clear and recreate Dirs
    for split in ["train", "test"]:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
        
    train_docs = {
        "doc_001.jpg": ("Chennai Tourism Guide", [
            "Chennai Tourism Guide",
            "Visit Marina Beach, the longest natural urban beach.",
            "Explore the ancient Kapaleeshwarar Temple in Mylapore."
        ]),
        "doc_002.jpg": ("Tamil Nadu Festival Calendar", [
            "Tamil Nadu Festival Calendar",
            "Pongal is a four-day harvest festival celebrated in January.",
            "People prepare sweet Pongal dish to offer to the Sun God."
        ]),
        "doc_003.jpg": ("Dravidian Architecture and Maps", [
            "Dravidian Architecture and Maps",
            "Temples are known for their towering gopurams (gateways).",
            "The Brihadisvara Temple at Thanjavur is a prime example."
        ]),
        "doc_004.jpg": ("Indian Geography", [
            "Indian Geography and Terrain",
            "The Himalayas stretch across the northern border.",
            "The Western Ghats dictate monsoon patterns in the south."
        ])
    }
    
    test_docs = {
        "test_001.jpg": ("Classical Dance Forms of India", [
            "Classical Dance Forms of India",
            "Bharatanatyam originated in the temples of Tamil Nadu.",
            "Kathakali is distinguished by elaborate makeup and costumes."
        ]),
        "test_002.jpg": ("Information about Marina Beach and Chennai highlights.", [
             "Chennai Coastal Life",
             "Marina Beach attracts thousands of tourists daily.",
             "The lighthouse provides a panoramic view of the bay."
        ])
    }
    
    train_mapping = {}
    for filename, (query, lines) in train_docs.items():
        out_path = os.path.join(base_dir, "train", filename)
        create_dumb_image(lines, out_path)
        train_mapping[filename] = query
        
    test_mapping = {}
    for filename, (query, lines) in test_docs.items():
        out_path = os.path.join(base_dir, "test", filename)
        create_dumb_image(lines, out_path)
        test_mapping[filename] = query
        
    with open(os.path.join(base_dir, "train_mapping.json"), "w") as f:
        json.dump(train_mapping, f, indent=4)
        
    with open(os.path.join(base_dir, "test_mapping.json"), "w") as f:
        json.dump(test_mapping, f, indent=4)
        
    print(f"Dataset generated at {base_dir}/train and {base_dir}/test")

if __name__ == "__main__":
    generate_samples()
