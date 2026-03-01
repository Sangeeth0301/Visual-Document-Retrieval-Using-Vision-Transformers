import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_attention_map(image_path, attention_weights, patch_size=16, img_size=224, save_dir="results/attention_maps"):
    """
    Overlays Transformer attention weights on the original PIL image.
    Saves the specific attention distribution maps into the evaluation folder.
    """
    from PIL import Image
    original_image = Image.open(image_path).convert("RGB")
    filename = os.path.basename(image_path)
    
    # Assuming batch size of 1 for visualization
    attention = attention_weights[0] # (num_heads, N+1, N+1)
    
    # Average the attention weights across all multi-head heads to get the total holistic focus
    avg_attention = attention.mean(dim=0) # (N+1, N+1)
    
    # Extract the [CLS] token's attention (index 0) to all actual image patches (indices 1 to 196)
    cls_attention = avg_attention[0, 1:] # Shape (196,)
    
    # Reshape the 196 sequentially flattened patches back into a 2D spatial grid (14x14)
    grid_size = img_size // patch_size
    attention_map = cls_attention.reshape(grid_size, grid_size).detach().cpu().numpy()
    
    # Normalize the weights so they map linearly to colors between 0 and 1
    attention_map = attention_map - np.min(attention_map)
    if np.max(attention_map) != 0:
        attention_map = attention_map / np.max(attention_map)
    
    # Resize the 14x14 heatmap spatially up to the exact pixel resolution of the original image (224x224)
    attention_map_resized = cv2.resize(attention_map, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    original_image = original_image.resize((img_size, img_size))
    img_np = np.array(original_image)
    
    # Plotting Logic
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img_np)
    ax[0].set_title("Original Document")
    ax[0].axis("off")
    
    # Overlay the heatmap on the original image
    ax[1].imshow(img_np)
    # The Jet colormap maps low attention to blue, high attention to dark red
    heatmap = ax[1].imshow(attention_map_resized, cmap='jet', alpha=0.5)
    ax[1].set_title(f"Self-Attention Heatmap Focus (Layer {len(attention_weights)})")
    ax[1].axis("off")
    
    plt.colorbar(heatmap, fraction=0.046, pad=0.04, ax=ax[1])
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"attn_{filename}")
    plt.savefig(out_path)
    plt.close()
    
    print(f"    Saved attention heatmap: {out_path}")

def save_retrieval_example(query_text, retrieved_images, scores, ranks, query_idx, save_dir="results/retrieval_examples"):
    """
    Saves visual side-by-side examples of a given text query and its Top-K retrieved documents.
    """
    os.makedirs(save_dir, exist_ok=True)
    from PIL import Image
    num_retrieved = len(retrieved_images)
    
    fig, axes = plt.subplots(1, num_retrieved, figsize=(4*num_retrieved, 4))
    fig.suptitle(f"Query: '{query_text}'", fontsize=16)
    
    if num_retrieved == 1:
        axes = [axes]
        
    for i in range(num_retrieved):
        img_path = retrieved_images[i]
        img = Image.open(img_path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"Rank {ranks[i]} (Sim: {scores[i]:.4f})\n{os.path.basename(img_path)}")
        axes[i].axis("off")
        
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"query_{query_idx}_results.png")
    plt.savefig(out_path)
    plt.close()
    
    print(f"    Saved retrieval visualization: {out_path}")
