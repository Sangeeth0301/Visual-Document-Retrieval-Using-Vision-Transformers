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
    
    # MiniLM text encoder is kept frozen for this educational exercise
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass: Images -> (Batch, 384)
        # Note: We must re-encode and train the graph, not use detached features.
        batch_tensors = [retriever.image_transform(img) for img in image_list]
        x = torch.stack(batch_tensors).to(retriever.device)
        
        vit_embeddings = retriever.vit_model(x)
        aligned_embeddings = retriever.proj(vit_embeddings)
        img_embs = F.normalize(aligned_embeddings, p=2, dim=-1)
        
        # 2. Forward Pass: Text Queries (frozen, detached)
        with torch.no_grad():
            text_raw = retriever.text_encoder.encode(queries_list, convert_to_tensor=True, device=retriever.device)
            text_embs = F.normalize(text_raw, p=2, dim=-1)
        
        # 3. Compute InfoNCE Loss
        loss = contrastive_loss(img_embs, text_embs)
        
        # 4. Backward & Step
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Contrastive Loss: {loss.item():.4f}")
            
    # Swap back to eval for inference
    retriever.vit_model.eval()
    return retriever
