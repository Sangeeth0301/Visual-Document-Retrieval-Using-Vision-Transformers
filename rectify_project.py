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

def run_rectified_experiment(name, model_type="better", pretrained=True, epochs=40):
    print(f"\n{'='*60}")
    print(f" RECTIFIED EXPERIMENT: {name}")
    print(f" Model: {model_type.upper()}, Pretrained: {pretrained}")
    print(f"{'='*60}")
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = "data/rectified_docs"
    train_json = "data/rectified_train_mapping.json"
    test_dir = "data/rectified_docs"
    test_json = "data/rectified_test_mapping.json"
    
    # Use augmentation for training
    train_loader = get_dataloader(train_dir, train_json, batch_size=4, shuffle=True, augment=True)
    test_loader = get_dataloader(test_dir, test_json, batch_size=4, shuffle=False, augment=False)
    
    if model_type == "better":
        vit = BetterViTEncoder(pretrained=pretrained)
    else:
        # Educational model
        vit = ViTEncoder(img_size=224, patch_size=16, embed_dim=768, depth=4, num_heads=8)
        
    retriever = DocumentRetriever(vit, device=device)
    
    output_dir = f"results/rectified_{name.replace(' ', '_').lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a slightly complex scheduler or just train for more epochs
    trained_retriever = train_retriever(retriever, train_loader, epochs=epochs, lr=5e-4, output_dir=output_dir)
    
    # Final Evaluation
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
    
    # Metrics
    metrics = compute_retrieval_metrics(query_embs_mat, doc_embs_mat, top_k_list=[1, 5])
    metrics["Experiment"] = name
    
    return metrics

def main():
    # Only two models needed for this specific rectification proof
    experiments = [
        {"name": "Scratch Model (Baseline)", "type": "educational", "pretrained": False},
        {"name": "Pretrained Model (Rectified)", "type": "better", "pretrained": True},
    ]
    
    results = []
    for exp in experiments:
        res = run_rectified_experiment(
            exp["name"], 
            model_type=exp["type"], 
            pretrained=exp["pretrained"],
            epochs=30
        )
        results.append(res)
        
    # Aggregate and Save
    df = pd.DataFrame(results)
    df.to_csv("results/rectified_comparison.csv", index=False)
    
    print("\n" + "="*60)
    print(" RECTIFICATION PROJECT SUCCESS")
    print("="*60)
    print(df[["Experiment", "Recall@1", "Recall@5"]])
    print("="*60)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    df_plot = df.melt(id_vars="Experiment", value_vars=["Recall@1", "Recall@5"], var_name="Metric", value_name="Percentage")
    sns.barplot(x="Experiment", y="Percentage", hue="Metric", data=df_plot, palette="coolwarm")
    plt.title("Rectification Evaluation: Scratch vs Pretrained")
    plt.ylim(0, 110)
    plt.savefig("results/rectification_plot.png")
    print(f"\nFinal comparison plot saved to results/rectification_plot.png")

if __name__ == "__main__":
    main()
