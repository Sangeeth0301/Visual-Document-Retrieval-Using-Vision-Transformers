# Visual Document Retrieval Using Vision Transformers

A Deep Learning educational baseline project that implements a Vision Transformer (ViT) from scratch to perform visual document retrieval. Inspired by modern vision-language models like ColPali, this project treats PDF documents holistically as images to preserve document layout, charts, and spatial features, matching them against text queries using a custom, lightweight ViT.

## 1. Problem Statement

Traditional document search heavily relies on OCR (Optical Character Recognition) to extract text, convert it to embeddings, and perform semantic search. However, this approach completely loses crucial visual and layout information. For example:
- The spatial relationship between a chart and its caption.
- The 2D structure of tables.
- Visual cues like font sizing, bolding, or positional emphasis.
- OCR errors propagate down the pipeline, ruining search results.

**The Core Challenge:** How can Transformers process PDF documents holistically using global self-attention while preserving their 2D spatial relationships?

## 2. Methodology Overview

This project bypasses OCR entirely. We adopt an image-centric approach:
1. **Visual Ingestion**: PDF pages are directly converted into high-resolution images.
2. **Patch Embeddings**: The image is broken down into 16x16 patches. A trainable linear projection maps these raw pixel patches into dense embeddings.
3. **Positional Encodings**: To prevent the Transformer from treating the patches as a disordered "bag of words," we inject 1D positional encodings, allowing the model to inherently learn 2D layout distances.
4. **Self-Attention & ViT**: A computationally efficient 2-4 layer from-scratch Vision Transformer processes the sequence. Multi-Head Attention allows different heads to focus on different aspects (e.g., text, charts, spatial structure). The scaling factor (`1/sqrt(d_k)`) stabilizes gradient flow.
5. **Cross-Modal Retrieval**: We project the final `[CLS]` token embedding of the ViT into the same vector space as a frozen lightweight text encoder (`MiniLM`). Documents are ranked against a user query using pure Cosine Similarity.
6. **Interpretability**: By extracting the Multi-Head Attention weights from the final Transformer layer, we overlay a heatmap on the original document to visually interpret exactly *where* the model was looking to satisfy a given query.

## 3. Dataset Description

The system processes a deterministic, synthetic visual document dataset generated on the fly.
* **Train / Test Split:** Evaluated documents are split into isolated `/data/train` and `/data/test` subsets.
* **Mappings:** Query-to-Image relations are explicitly mapped in deterministic JSON files (`train_mapping.json`, `test_mapping.json`). 
* **Preprocessing:** Images are deterministically resized to `224x224`, converted to tensors, and Z-score normalized against ImageNet specifications using PyTorch primitives.

## 4. Experimental Setup

*   **ViT Architecture**: 
    *   **Patch Size:** 16x16
    *   **Embedding Dimension:** 768D (projected down to 384D for Text alignment)
    *   **Layers:** 2 Transformer Layers
    *   **Attention Heads:** 4 parallel heads
    *   **Regularization:** Dropout at $p=0.1$ across MLP and Attention projections. Learnable parameters initialized using Truncated Normal distribution.
*   **Optimization**:
    *   **Optimizer:** `AdamW`
    *   **Learning Rate:** $1e-3$ for Projection arrays, $1e-4$ for ViT parameters.
    *   **Loss Function:** Symmetric InfoNCE Contrastive Loss (Temperature = $0.07$).
    *   **Compute Target:** Operates efficiently on standard CPU or Google Colab environments.

## 5. Evaluation Metrics

Because our system natively matches queries to documents, we utilize standard retrieval evaluation statistics computed over cosine similarity score spaces.
*   **Recall@1**: The percentage of test queries for which the exact matching document is ranked absolutely first.
*   **Recall@5**: The percentage of test queries for which the exact matching document appears anywhere within the top 5 ranked documents. Highly useful in general search engines.

## 6. Quantitative Results

Evaluation is fully automated post-training. The resulting accuracy arrays are written to the `/results/metrics.csv` evaluation artifact.

| Model | Evaluation Split | Recall@1 | Recall@5 |
|-------|------------------|----------|----------|
| Custom Baseline ViT | Test | Computed at Runtime | Computed at Runtime |

> *Note: Due to the micro-batch size utilized for educational baselining purposes, metrics rapidly approach 100% post-convergence. A training loss plot is generated dynamically at `/results/training_curves.png`.*

## 7. Attention Visualization

Inside the `/results/attention_maps/` directory, the framework isolates the structural probability layout emitted by the final multi-head block via the `[CLS]` token. 
Dark red hotspots highlight the precise patches the Transformer focused on structurally to answer the given text query without OCR. Examples of query-to-document mappings can be found in `/results/retrieval_examples/`.

## 8. Limitations

*   **Small Dataset Scale**: Due to CPU overhead, the dataset scale is artificially compressed. Massive datasets would necessitate `DistributedDataParallel` implementation and memory optimizations.
*   **1D Positional Encodings**: The 1D sequence mapping does not distinctly capture absolute Cartesian distances in horizontal vs vertical document layouts inherently.
*   **Frozen Language Embeddings**: The text representation manifold remains static due to freezing `MiniLM`, limiting bi-modal alignment ceiling capabilities.

## 9. Future Work

While 1D positional encoding preserves patch ordering, future work may explore 2D relative positional embeddings for improved spatial modeling.

## 10. Reproducibility

This baseline is fully operational within local Anaconda/Virtual Environments and completely compatible with **Google Colab**.

### Installation
```bash
git clone https://github.com/Sangeeth0301/Visual-Document-Retrieval-Using-Vision-Transformers.git
cd Visual-Document-Retrieval-Using-Vision-Transformers
pip install -r requirements.txt
```

### Execution
The framework initializes deterministic random seeds internally. Running the primary orchestrator automatically builds the datasets via the `create_dataset.py` pipeline, triggers Contrastive Training, processes Metric Evaluation, and writes all plotting visualizations to the `/results` directory.

```bash
python main.py
```
