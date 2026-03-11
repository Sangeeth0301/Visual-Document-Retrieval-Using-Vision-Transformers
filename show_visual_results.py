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

def show_result():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the best model
    # Note: Since the previous training run just completed, we can't rely on a saved file being 
    # in a specific spot without re-running or knowing the exact output path.
    # I'll just re-initialize the pretrained model which has the weights we want.
    vit = BetterViTEncoder(pretrained=True)
    retriever = DocumentRetriever(vit, device=device)
    
    # Data
    test_dir = "data/rectified_docs"
    test_json = "data/rectified_test_mapping.json"
    test_loader = get_dataloader(test_dir, test_json, batch_size=9, shuffle=False)
    
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    queries = batch["query"]
    paths = batch["image_path"]
    
    # Embed everything
    vit.eval()
    with torch.no_grad():
        doc_embs = F.normalize(retriever.proj(vit(images)), p=2, dim=-1)
        query_embs = retriever.embed_queries(queries)
    
    # Pick the 4th document (Gamma Rays paper)
    q_idx = 3 # index 3
    query_text = queries[q_idx]
    q_vec = query_embs[q_idx].unsqueeze(0)
    
    # Search
    # We compare the query vector against ALL doc_embs in the batch
    sims = torch.mm(q_vec, doc_embs.t()).squeeze(0)
    scores, indices = torch.topk(sims, k=3)
    
    ret_paths = [paths[idx] for idx in indices]
    ret_scores = scores.cpu().numpy()
    ranks = [1, 2, 3]
    
    # Save the visual result
    save_dir = "final_visual_results"
    os.makedirs(save_dir, exist_ok=True)
    save_retrieval_example(query_text, ret_paths, ret_scores, ranks, "final_real", save_dir=save_dir)
    
    print(f"Final Visual Result saved to {save_dir}/query_final_real_results.png")

if __name__ == "__main__":
    show_result()
