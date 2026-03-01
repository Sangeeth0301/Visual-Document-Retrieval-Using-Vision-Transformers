import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_attention_map(original_image, attention_weights, patch_size=16, img_size=224):
    """
    Overlays Transformer attention weights on the original PIL image.
    original_image: PIL Image
    attention_weights: tensor from the last Transformer layer, shape (Batch, num_heads, sequence_length, sequence_length)
    """
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
    # We use INTER_CUBIC for smooth visual gradients
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
    ax[1].set_title("Self-Attention Heatmap Focus")
    ax[1].axis("off")
    
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return fig
