import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torchvision.transforms as transforms

class DocumentRetriever:
    def __init__(self, vit_model, device="cpu"):
        self.device = device
        self.vit_model = vit_model.to(self.device).eval()
        
        # Lightweight text encoder for embedding user queries (generates 384D vectors)
        print("Loading MiniLM Text Encoder...")
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.text_encoder.eval()
        
        self.text_dim = 384
        self.vit_dim = 768
        
        # Projection layer: Aligns the 768D Vision representation to the 384D Text representation
        self.proj = torch.nn.Linear(self.vit_dim, self.text_dim).to(self.device)
        
        # Standard ViT transformations
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def embed_images(self, pil_images):
        """Pass a list of PIL images through the custom ViT to obtain embeddings."""
        batch_tensors = []
        for img in pil_images:
            tensor = self.image_transform(img)
            batch_tensors.append(tensor)
            
        x = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            vit_embeddings = self.vit_model(x) # (Batch, 768)
            
            # Since our Text Model is pre-trained and our ViT is scratch-built,
            # we project ViT's space onto the text space to match dimensions (768 -> 384)
            aligned_embeddings = self.proj(vit_embeddings)
            
        # L2 Normalize for Cosine Similarity
        return F.normalize(aligned_embeddings, p=2, dim=-1)

    def embed_queries(self, queries):
        """Pass a list of text queries through MiniLM."""
        with torch.no_grad():
            # Encodes text into (Batch, 384)
            text_embeddings = self.text_encoder.encode(queries, convert_to_tensor=True, device=self.device)
        
        # L2 Normalize for Cosine Similarity
        return F.normalize(text_embeddings, p=2, dim=-1)

    def search(self, query_embeddings, doc_embeddings, top_k=3):
        """
        Computes the cosine similarity between the queries and documents.
        Because both embeddings are L2 normalized, dot product == cosine similarity.
        """
        # query_embeddings: (Q, D)
        # doc_embeddings: (N, D)
        # Output similarities: matrix of (Q, N)
        similarities = torch.matmul(query_embeddings, doc_embeddings.T)
        
        # Retrieve the Top-K most similar documents per query
        max_k = min(top_k, similarities.shape[1])
        top_k_scores, top_k_indices = torch.topk(similarities, max_k, dim=1)
        
        return top_k_scores, top_k_indices
