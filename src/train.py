import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Computes InfoNCE Contrastive Loss to align image and text embeddings.
    Since we know the exact 1:1 image to query match, the ground truth is the identity matrix.
    """
    # Compute similarity between all pairs in the batch
    # Shape: (B, B)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Target is the diagonal: image i matches text i
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(image_embeddings.device)
    
    # Loss calculated symmetrically over rows (images) and columns (texts)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def train_retriever(retriever, image_list, queries_list, epochs=30, lr=5e-4):
    """
    Trains the ViT cross-modal projection layer and fine-tunes the Visformer
    to overfit a tiny visual document dataset for our baseline mini-task.
    """
    retriever.vit_model.train()
    print(f"Starting Contrastive Alignment Training for {epochs} epochs...")
    
    optimizer = optim.AdamW([
        {'params': retriever.vit_model.parameters(), 'lr': lr * 0.1}, # Fine-tune ViT slowly
        {'params': retriever.proj.parameters(), 'lr': lr}             # Train projection layer faster
    ])
    
    # Pre-compute fixed tensors to eliminate major pipeline lags
    print("Pre-computing text embeddings and image vectors to avoid training lags...")
    with torch.no_grad():
        text_raw = retriever.text_encoder.encode(queries_list, convert_to_tensor=True, device=retriever.device)
        text_embs = F.normalize(text_raw, p=2, dim=-1)
        
    batch_tensors = [retriever.image_transform(img) for img in image_list]
    x = torch.stack(batch_tensors).to(retriever.device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass: Images -> (Batch, 384)
        vit_embeddings = retriever.vit_model(x)
        aligned_embeddings = retriever.proj(vit_embeddings)
        img_embs = F.normalize(aligned_embeddings, p=2, dim=-1)
        
        # 2. Compute InfoNCE Loss vs pre-computed text queries
        loss = contrastive_loss(img_embs, text_embs)
        
        # Rapid Rank-1 Accuracy Tracking
        with torch.no_grad():
            sims = torch.matmul(img_embs, text_embs.T)
            preds = torch.argmax(sims, dim=1)
            targets = torch.arange(len(preds)).to(retriever.device)
            acc = torch.sum(preds == targets).float() / len(preds) * 100.0
        
        # 3. Backward & Step
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Contrastive Loss: {loss.item():.4f} - Exact Batch Match: {acc:.1f}%")
            
    # Swap back to eval for inference
    retriever.vit_model.eval()
    return retriever
