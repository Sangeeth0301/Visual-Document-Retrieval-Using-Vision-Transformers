import os
import torch
import warnings

# Suppress HuggingFace and local warnings to clean up output
warnings.filterwarnings("ignore") 

from src.dataset_loader import get_dataloader
from src.model import ViTEncoder
from src.retrieval import DocumentRetriever
from src.train import train_retriever, set_seed
from src.evaluate import compute_retrieval_metrics, save_metrics
from src.visualize import plot_attention_map, save_retrieval_example

def main():
    print("\n" + "="*60)
    print(" VISUAL DOCUMENT RETRIEVAL - EVALUATION FRAMEWORK")
    print("="*60)
    
    # 0. Set Reproducibility
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware backend: {device.upper()}")
    
    # 1. Ensure Data Exists
    train_dir = "data/train"
    train_json = "data/train_mapping.json"
    test_dir = "data/test"
    test_json = "data/test_mapping.json"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_json):
        print("--> Generating deterministic Document Dataset splits...")
        import create_dataset
        create_dataset.generate_samples()
        
    print(f"\n[1] Initializing PyTorch DataLoaders...")
    train_loader = get_dataloader(train_dir, train_json, batch_size=2, shuffle=True)
    test_loader = get_dataloader(test_dir, test_json, batch_size=2, shuffle=False)
    
    # 2. Instantiate Model
    print("\n[2] Instantiating Vision Transformer (ViT)...")
    vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=768, depth=2, num_heads=4, drop_rate=0.1, attn_drop_rate=0.1)
    retriever = DocumentRetriever(vit, device=device)
    
    # 3. Train
    print("\n[3] Executing InfoNCE Contrastive Training across Dataloader...")
    trained_retriever = train_retriever(retriever, train_loader, epochs=40, lr=1e-3, output_dir="results")
    
    # 4. Evaluation Vectorization
    print("\n[4] Computing Inference and Metrics on Test Set...")
    all_doc_embs = []
    all_query_embs = []
    all_doc_paths = []
    
    for batch in test_loader:
        images = batch["image"].to(device)
        queries = batch["query"]
        paths = batch["image_path"]
        
        # Images
        with torch.no_grad():
            vit_embeddings = trained_retriever.vit_model(images)
            aligned_embeddings = trained_retriever.proj(vit_embeddings)
            img_embs = torch.nn.functional.normalize(aligned_embeddings, p=2, dim=-1)
            
            # Text Tracking
            text_raw = trained_retriever.text_encoder.encode(queries, convert_to_tensor=True, device=device)
            text_embs = torch.nn.functional.normalize(text_raw, p=2, dim=-1)
            
            all_doc_embs.append(img_embs)
            all_query_embs.append(text_embs)
            all_doc_paths.extend(paths)
            
    doc_embs_mat = torch.cat(all_doc_embs)
    query_embs_mat = torch.cat(all_query_embs)
    
    # 5. Compute Quantitative Metrics
    metrics = compute_retrieval_metrics(query_embs_mat, doc_embs_mat, top_k_list=[1, 5])
    save_metrics(metrics, output_file="results/metrics.csv")
    
    # 6. Qualitative Results (Top-1 Visualizations)
    print("\n[6] Rendering Visual Examples to `/results/`...")
    top_k = min(2, len(all_doc_paths))
    sims = torch.matmul(query_embs_mat, doc_embs_mat.T)
    top_scores, top_indices = torch.topk(sims, top_k, dim=1)
    
    for i in range(len(all_doc_paths)):
        retrieved_paths = [all_doc_paths[idx.item()] for idx in top_indices[i]]
        save_retrieval_example(
            query_text=test_loader.dataset.queries[i],
            retrieved_images=retrieved_paths,
            scores=top_scores[i].tolist(),
            ranks=list(range(1, top_k + 1)),
            query_idx=i,
            save_dir="results/retrieval_examples"
        )
        
        # Heatmap generation for Top 1 retrieved doc
        best_doc_idx = top_indices[i][0]
        best_image = test_loader.dataset[best_doc_idx.item()]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
             _, attn_weights = trained_retriever.vit_model(best_image, return_attn=True)
             
        plot_attention_map(
            image_path=retrieved_paths[0], 
            attention_weights=attn_weights[-1], 
            save_dir="results/attention_maps"
        )
        
    print("\nExperiment Run Complete! Check the /results/ folder for outputs.")

if __name__ == "__main__":
    main()
