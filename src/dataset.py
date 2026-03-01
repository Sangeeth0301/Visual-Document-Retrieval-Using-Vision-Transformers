import os
import fitz # PyMuPDF
from PIL import Image
import io

def pdf_to_pages(pdf_path, dpi=150, target_size=(224, 224)):
    """Convert a PDF file into a list of resized PIL Images using PyMuPDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find PDF: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    resized_pages = []
    
    # Calculate zoom factor for equivalent DPI (PyMuPDF default is 72)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize to standard input size for ViT
        page_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_pages.append(page_resized)
        
    return resized_pages

def load_documents(pdf_directory, target_size=(224, 224)):
    """Loads all PDFs from a directory and returns a flat list of page images and metadata."""
    corpus_images = []
    metadata = [] # Stores dicts with filename and page_num
    
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory: {pdf_directory}. Please place your PDF documents here.")
        return corpus_images, metadata
        
    for filename in sorted(os.listdir(pdf_directory)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            pages = pdf_to_pages(pdf_path, dpi=150, target_size=target_size)
            for i, page in enumerate(pages):
                corpus_images.append(page)
                metadata.append({"file": filename, "page": i + 1})
                
    return corpus_images, metadata
