import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset_loader import get_dataloader
from src.model import ViTEncoder
from src.retrieval import DocumentRetriever
from src.train import train_retriever, set_seed
from src.evaluate import compute_retrieval_metrics, save_metrics
import warnings

warnings.filterwarnings("ignore")

def run_experiment(name, num_heads=4, use_pos_embed=True, use_scaling=True, epochs=30):
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: {name}")
    print(f" Heads: {num_heads}, PosEmbed: {use_pos_embed}, Scaling: {use_scaling}")
    print(f"{'='*60}")
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = "data/real_docs"
    train_json = "data/real_train_mapping.json"
    test_dir = "data/real_docs"
    test_json = "data/real_test_mapping.json"
    
    train_loader = get_dataloader(train_dir, train_json, batch_size=2, shuffle=True)
    test_loader = get_dataloader(test_dir, test_json, batch_size=2, shuffle=False)
    
    vit = ViTEncoder(
        img_size=224, patch_size=16, embed_dim=768, depth=2, 
        num_heads=num_heads, drop_rate=0.1, attn_drop_rate=0.1,
        use_pos_embed=use_pos_embed, use_scaling=use_scaling
    )
    retriever = DocumentRetriever(vit, device=device)
    
    output_dir = f"results/{name.replace(' ', '_').lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    trained_retriever = train_retriever(retriever, train_loader, epochs=epochs, lr=1e-3, output_dir=output_dir)
    
    # Inference on Test Set
    all_doc_embs = []
    all_query_embs = []
    
    trained_retriever.vit_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            queries = batch["query"]
            
            # Use the same logic as training for alignment
            vit_embeddings = trained_retriever.vit_model(images)
            aligned_embeddings = trained_retriever.proj(vit_embeddings)
            img_embs = F.normalize(aligned_embeddings, p=2, dim=-1)
            
            # Text Tracking
            text_embs = trained_retriever.embed_queries(queries)
            
            all_doc_embs.append(img_embs)
            all_query_embs.append(text_embs)
            
    doc_embs_mat = torch.cat(all_doc_embs)
    query_embs_mat = torch.cat(all_query_embs)
    
    metrics = compute_retrieval_metrics(query_embs_mat, doc_embs_mat, top_k_list=[1, 3])
    metrics["Experiment"] = name
    
    return metrics

def main():
    experiments = [
        {"name": "Multi-Head (Optimal)", "num_heads": 4, "use_pos_embed": True, "use_scaling": True},
        {"name": "Self-Attention (1 Head)", "num_heads": 1, "use_pos_embed": True, "use_scaling": True},
        {"name": "No Positional Encoding", "num_heads": 4, "use_pos_embed": False, "use_scaling": True},
        {"name": "No Scaling Factor", "num_heads": 4, "use_pos_embed": True, "use_scaling": False},
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment(
            exp["name"], 
            num_heads=exp["num_heads"], 
            use_pos_embed=exp["use_pos_embed"], 
            use_scaling=exp["use_scaling"],
            epochs=30
        )
        results.append(res)
        
    # Aggregate and Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/comparison_results.csv", index=False)
    
    print("\n" + "="*60)
    print(" COMPARATIVE SUMMARY")
    print("="*60)
    print(df[["Experiment", "Recall@1", "Recall@3"]])
    print("="*60)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot Recall@1
    plt.subplot(1, 2, 1)
    sns.barplot(x="Experiment", y="Recall@1", data=df, palette="viridis")
    plt.title("Recall @ 1 Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, 110)
    
    # Plot Recall@3
    plt.subplot(1, 2, 2)
    sns.barplot(x="Experiment", y="Recall@3", data=df, palette="magma")
    plt.title("Recall @ 3 Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig("results/comparison_plot.png")
    print(f"\nSaved comparison plot to results/comparison_plot.png")

if __name__ == "__main__":
    main()
