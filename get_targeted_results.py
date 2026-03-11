import os
import torch
import torch.nn.functional as F
from PIL import Image
from src.dataset_loader import get_dataloader
from src.model import BetterViTEncoder
from src.retrieval import DocumentRetriever
from src.visualize import save_retrieval_example
import warnings

warnings.filterwarnings("ignore")

def show_specific_results():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model and Data
    vit = BetterViTEncoder(pretrained=True)
    retriever = DocumentRetriever(vit, device=device)
    
    test_dir = "data/rectified_docs"
    test_json = "data/rectified_test_mapping.json"
    test_loader = get_dataloader(test_dir, test_json, batch_size=20, shuffle=False)
    
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    queries = batch["query"]
    paths = batch["image_path"]
    
    # 2. Embed
    vit.eval()
    with torch.no_grad():
        doc_embs = F.normalize(retriever.proj(vit(images)), p=2, dim=-1)
        query_embs = retriever.embed_queries(queries)
    
    # 3. Targeted Indices
    # Looking at rectified_test_mapping.json:
    # doc 34 is at index 0
    # doc 5  is at index 4
    # doc 6  is at index 5
    target_indices = [0, 4, 5]
    
    save_dir = "specific_results_34_5_6"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nRetrieval Results for Specific Documents:")
    print("-" * 50)
    
    results_summary = []
    
    for q_idx in target_indices:
        query_text = queries[q_idx]
        q_vec = query_embs[q_idx].unsqueeze(0)
        
        # Search
        sims = torch.mm(q_vec, doc_embs.t()).squeeze(0)
        scores, indices = torch.topk(sims, k=3)
        
        ret_paths = [paths[idx] for idx in indices]
        ret_scores = scores.cpu().numpy()
        ranks = [1, 2, 3]
        
        doc_name = os.path.basename(paths[q_idx])
        print(f"Document: {doc_name}")
        print(f"Query: {query_text}")
        print(f"Top Result: {os.path.basename(ret_paths[0])} (Score: {ret_scores[0]:.4f})")
        print("-" * 50)
        
        # Save visual comparison
        save_retrieval_example(query_text, ret_paths, ret_scores, ranks, doc_name.split('.')[0], save_dir=save_dir)
        results_summary.append({
            "Doc": doc_name,
            "Query": query_text,
            "Top_Match": os.path.basename(ret_paths[0]),
            "Score": ret_scores[0]
        })

    print(f"\nVisual results saved to: {save_dir}/")

if __name__ == "__main__":
    show_specific_results()
