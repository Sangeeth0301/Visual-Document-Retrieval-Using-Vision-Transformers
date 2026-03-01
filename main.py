import os
import torch
import warnings

# Suppress HuggingFace and local warnings to clean up output
warnings.filterwarnings("ignore") 

from src.dataset import load_documents
from src.model import ViTEncoder
from src.retrieval import DocumentRetriever
from src.train import train_retriever
from src.visualize import plot_attention_map

def main():
    print("\n" + "="*50)
    print(" EDUCATIONAL VISUAL DOCUMENT RETRIEVAL BASELINE")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware backend: {device.upper()}")
    
    # 1. Load Data
    data_dir = "data/pdfs"
    print(f"\n[1] Attempting to load PDFs from {data_dir}...")
    corpus_images, metadata = load_documents(data_dir)
    
    if not corpus_images:
        print(f"--> No PDFs found. Please generate them first by running: python create_dummy_pdfs.py")
        return
        
    print(f"--> Successfully loaded {len(corpus_images)} document pages.")
    
    # Define our "Mini-Task" Training target queries
    # Since we are training a ViT directly over a small synthetic corpus, 
    # we pair each page with a descriptive matching query.
    training_queries = [
        "A travel guide detailing Marina Beach, Kapaleeshwarar Temple, and Chennai tourism highlights.",
        "Tamil Nadu festival dates, specifically information regarding the Pongal harvest celebration.",
        "Dravidian temple architecture, gopurams, and Thanjavur carvings."
    ]
    
    if len(training_queries) != len(corpus_images):
        print("--> Warning: Training queries do not have a 1:1 match with pages. Some documents may not be learned.")
        
    # 2. Instantiate Model
    print("\n[2] Instantiating the scratch-built Vision Transformer...")
    # A tiny ViT tailored for this baseline 
    vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=768, depth=2, num_heads=4)
    retriever = DocumentRetriever(vit, device=device)
    
    # 3. Train the System (Contrastive Image-Text Alignment)
    print("\n[3] Initiating Contrastive Alignment Training...")
    print("    NOTE: An untrained ViT outputs random structural noise.")
    print("    By training on our mini-task, we functionally link visual structures")
    print("    to their semantic representations via cosine projection.")
    
    # Train for 40 epochs to overfit our small 3-document dataset
    trained_retriever = train_retriever(retriever, corpus_images, training_queries, epochs=40, lr=1e-3)
    
    # 4. Inference Phase
    print("\n[4] Execution: Visual Document Retrieval Simulation")
    
    # Try a query conceptually similar to document 1 (Chennai beach)
    user_query = "Where can I find information about the longest natural urban beach in Chennai?"
    print(f"--> User Query: '{user_query}'")
    
    # Embed the corpus images with our freshly trained ViT
    doc_embs = trained_retriever.embed_images(corpus_images)
    
    # Embed the text query using MiniLM
    query_emb = trained_retriever.embed_queries([user_query])
    
    # Search for matching layouts using Cosine Similarity
    scores, indices = trained_retriever.search(query_emb, doc_embs, top_k=2)
    
    print("\n--- Retrieval Results ---")
    best_match_idx = indices[0][0].item()
    for i in range(len(indices[0])):
        idx = indices[0][i].item()
        score = scores[0][i].item()
        doc_meta = metadata[idx]
        print(f"Rank {i+1}: {doc_meta['file']} (Page {doc_meta['page']}) -> Sim Score: {score:.4f}")
        
    # 5. Extract and Plot the Attention Map
    print(f"\n[5] Extracting Multi-Head Attention Layout from the Top Result...")
    best_image = corpus_images[best_match_idx]
    
    # Convert image back to a tensor to pass through the ViT one last time and grab the weights
    img_tensor = trained_retriever.image_transform(best_image).unsqueeze(0).to(device)
    
    # We must explicitly request the attentions from the forward loop
    with torch.no_grad():
        _, attn_weights = trained_retriever.vit_model(img_tensor, return_attn=True)
        
    # Isolate the final Transformer block's attention matrix
    last_layer_attn = attn_weights[-1] 
    
    print("    Opening Heatmap visualization window. Close the window to exit the program.")
    plot_attention_map(best_image, last_layer_attn)
    print("\nExecution complete.")

if __name__ == "__main__":
    main()
