import os
import torch
import torch.nn.functional as F
from PIL import Image
from src.dataset_loader import get_dataloader
from src.model import BetterViTEncoder
from src.retrieval import DocumentRetriever
from src.train import train_retriever
from src.visualize import save_retrieval_example
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def produce_real_result():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    data_dir = "data/rectified_docs"
    mapping_json = "data/rectified_test_mapping.json"
    # Using the train mapping too so we have more docs to search from
    train_mapping = "data/rectified_train_mapping.json"
    
    train_loader = get_dataloader(data_dir, train_mapping, batch_size=4, shuffle=True, augment=True)
    test_loader = get_dataloader(data_dir, mapping_json, batch_size=10, shuffle=False)
    
    # 2. Get the model
    vit = BetterViTEncoder(pretrained=True)
    retriever = DocumentRetriever(vit, device=device)
    
    # 3. Simple Training (Fine-tuning for 3 epochs only to align vectors correctly for documents)
    # We don't need full training just for a logic proof
    trained_retriever = train_retriever(retriever, train_loader, epochs=5, lr=5e-4)
    
    # 4. Perform Retrieval on a specific test query
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    queries = batch["query"]
    paths = batch["image_path"]
    
    # Align the vector space
    vit.eval()
    with torch.no_grad():
        doc_embs = F.normalize(trained_retriever.proj(vit(images)), p=2, dim=-1)
        query_embs = trained_retriever.embed_queries(queries)
    
    # Query: 'Very High Energy Gamma Rays from Ultra Fast Outflows'
    q_idx = 3 # This is the index in our test set
    query_text = queries[q_idx]
    q_vec = query_embs[q_idx].unsqueeze(0)
    
    # Compute similarity against all docs in this batch
    sims = torch.mm(q_vec, doc_embs.t()).squeeze(0)
    scores, indices = torch.topk(sims, k=3)
    
    ret_paths = [paths[idx] for idx in indices]
    ret_scores = scores.cpu().numpy()
    ranks = [1, 2, 3]
    
    # 5. Save the visual evidence
    save_dir = "actual_results"
    os.makedirs(save_dir, exist_ok=True)
    save_retrieval_example(query_text, ret_paths, ret_scores, ranks, "live_demo", save_dir=save_dir)
    
    print(f"Visual Retrieval Output saved to {save_dir}/query_live_demo_results.png")

if __name__ == "__main__":
    produce_real_result()
