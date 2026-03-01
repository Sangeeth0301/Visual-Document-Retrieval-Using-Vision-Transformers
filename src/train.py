import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def set_seed(seed=42):
    """Ensure reproducibility by fixing the exact random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """InfoNCE Contrastive Loss to symmetrically align image layout logic to language query definitions."""
    # (B, B) similarity matrix
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Ground truth is identity (1:1 matching in batch)
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(image_embeddings.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def train_retriever(retriever, train_loader, epochs=30, lr=1e-3, output_dir="results"):
    set_seed(42)  # Maintain deterministic nature for evaluation
    retriever.vit_model.train()
    print(f"\nStarting Contrastive Alignment Training for {epochs} epochs...")
    
    optimizer = optim.AdamW([
        {'params': retriever.vit_model.parameters(), 'lr': lr * 0.1},
        {'params': retriever.proj.parameters(), 'lr': lr}            
    ])
    
    epoch_losses = []
    epoch_accs = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            images = batch["image"].to(retriever.device)
            queries = batch["query"]
            
            # Forward Pass: Images
            vit_embeddings = retriever.vit_model(images)
            aligned_embeddings = retriever.proj(vit_embeddings)
            img_embs = F.normalize(aligned_embeddings, p=2, dim=-1)
            
            # Forward Pass: Text (Frozen)
            with torch.no_grad():
                text_raw = retriever.text_encoder.encode(queries, convert_to_tensor=True, device=retriever.device)
                text_embs = F.normalize(text_raw, p=2, dim=-1)
            
            # Contrastive Objective
            loss = contrastive_loss(img_embs, text_embs)
            
            # Metric Tracking
            with torch.no_grad():
                sims = torch.matmul(img_embs, text_embs.T)
                preds = torch.argmax(sims, dim=1)
                targets = torch.arange(len(preds)).to(retriever.device)
                acc = torch.sum(preds == targets).float() / len(preds) * 100.0
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Contrastive Loss: {avg_loss:.4f} - Exact Batch Match: {avg_acc:.1f}%")
            
    # Swap back to eval
    retriever.vit_model.eval()
    
    # Plotting Training Curves
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='b', label='Train Loss')
    plt.title('Contrastive Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('InfoNCE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    print(f"Training loop complete. Saved loss curves to {output_dir}/training_curves.png")
    
    return retriever
