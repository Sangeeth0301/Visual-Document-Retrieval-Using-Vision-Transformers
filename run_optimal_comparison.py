import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset_loader import get_dataloader
from src.model import ViTEncoder, BetterViTEncoder
from src.retrieval import DocumentRetriever
from src.train import train_retriever, set_seed
from src.evaluate import compute_retrieval_metrics
import warnings

warnings.filterwarnings("ignore")

def run_opt_experiment(name, num_heads=1, use_pretrained=True):
    print(f"\n{'='*60}")
    print(f" OPTIMAL RUN: {name} (Heads: {num_heads})")
    print(f"{'='*60}")
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = "data/rectified_docs"
    train_json = "data/rectified_train_mapping.json"
    test_dir = "data/rectified_docs"
    test_json = "data/rectified_test_mapping.json"
    
    train_loader = get_dataloader(train_dir, train_json, batch_size=4, shuffle=True, augment=True)
    test_loader = get_dataloader(test_dir, test_json, batch_size=4, shuffle=False)
    
    # We use BetterViTEncoder but we need to modify the model to support head count comparison
    # For a fair 'optimal' test, we'll use a custom version of our ViT with the same depth
    # but varying heads, and we'll train them equally.
    
    vit = ViTEncoder(
        img_size=224, patch_size=16, embed_dim=768, depth=4, 
        num_heads=num_heads, drop_rate=0.1, attn_drop_rate=0.1
    )
    
    retriever = DocumentRetriever(vit, device=device)
    
    output_dir = f"opt_results/{name.replace(' ', '_').lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Higher epochs for 'optimal' convergence
    trained_retriever = train_retriever(retriever, train_loader, epochs=40, lr=1e-3, output_dir=output_dir)
    
    # Eval
    trained_retriever.vit_model.eval()
    all_doc_embs = []
    all_query_embs = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            queries = batch["query"]
            
            vit_embeddings = trained_retriever.vit_model(images)
            aligned_embeddings = trained_retriever.proj(vit_embeddings)
            img_embs = F.normalize(aligned_embeddings, p=2, dim=-1)
            text_embs = trained_retriever.embed_queries(queries)
            
            all_doc_embs.append(img_embs)
            all_query_embs.append(text_embs)
            
    doc_embs_mat = torch.cat(all_doc_embs)
    query_embs_mat = torch.cat(all_query_embs)
    
    metrics = compute_retrieval_metrics(query_embs_mat, doc_embs_mat, top_k_list=[1, 3, 5])
    
    # Save individual report
    pd.DataFrame([metrics]).to_csv(f"{output_dir}/report.csv", index=False)
    
    metrics["Model"] = name
    return metrics

def main():
    os.makedirs("opt_results", exist_ok=True)
    
    results = []
    
    # Experiment 1: Pure Self-Attention (1 Head)
    results.append(run_opt_experiment("Self-Attention Transformer", num_heads=1))
    
    # Experiment 2: Multi-Head Transformer (8 Heads)
    results.append(run_opt_experiment("Multi-Head Transformer", num_heads=8))
    
    # Final Comparison
    df = pd.DataFrame(results)
    df.to_csv("opt_results/optimal_comparison.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    df_melted = df.melt(id_vars="Model", value_vars=["Recall@1", "Recall@3", "Recall@5"], var_name="Metric", value_name="Percentage")
    sns.barplot(x="Metric", y="Percentage", hue="Model", data=df_melted, palette="Set2")
    plt.title("Optimal Results: Self-Attention vs Multi-Head")
    plt.ylim(0, 110)
    plt.savefig("opt_results/final_comparison_chart.png")

if __name__ == "__main__":
    main()
