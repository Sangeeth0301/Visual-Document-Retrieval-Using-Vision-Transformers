import torch
import numpy as np
import os
import pandas as pd

def compute_retrieval_metrics(query_embeddings, doc_embeddings, top_k_list=[1, 5, 10]):
    """
    Computes Recall@K and Precision@K for a retrieval system.
    Assumes queries[i] matches exactly to doc[i] (1-to-1 mapping for the test set).
    """
    assert query_embeddings.shape[0] == doc_embeddings.shape[0], "Embeddings mismatch"
    N = query_embeddings.shape[0]
    
    # query_embeddings: (N, D), doc_embeddings: (M, D) (M=N here)
    # Cosine similarity matrix (N, N)
    similarities = torch.matmul(query_embeddings, doc_embeddings.T)
    
    # Sort indices in descending order
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    
    metrics = {}
    for k in top_k_list:
        max_k = min(k, N)
        top_k_indices = sorted_indices[:, :max_k]
        
        # Ground truth is the diagonal index [0, 1, 2... N-1]
        targets = torch.arange(N).view(-1, 1).expand(-1, max_k).to(doc_embeddings.device)
        
        # Check if the correct document is within the top-k retrieved
        hits = (top_k_indices == targets).any(dim=1).float()
        
        recall_k = hits.mean().item() * 100.0
        # For precision, it's 1 hit out of K retrieved documents
        precision_k = (hits / max_k).mean().item() * 100.0
        
        metrics[f'Recall@{k}'] = recall_k
        metrics[f'Precision@{k}'] = precision_k
        
    return metrics

def save_metrics(metrics, output_file="results/metrics.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(output_file, index=False)
    print(f"\nSaved evaluation metrics to {output_file}")
    
    print("-" * 50)
    print(" EVALUATION METRICS")
    print("-" * 50)
    for k, v in metrics.items():
        print(f" {k:15}: {v:.2f}%")
    print("-" * 50)
