import json
import os

mapping_all = {
  "rectified_doc_1.png": "FVG-PT: Adaptive Foreground View-Guided Prompt Tuning for Vision-Language Models",
  "rectified_doc_2.png": "Impermanent: A Live Benchmark for Temporal Generalization in Time Series Forecasting",
  "rectified_doc_3.png": "HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising",
  "rectified_doc_4.png": "Very High Energy Gamma Rays from Ultra Fast Outflows",
  "rectified_doc_5.png": "Improved Certificates for Independence Number in Semirandom Hypergraphs",
  "rectified_doc_6.png": "Fermi-pressure-assisted cavity superradiance in a mesoscopic Fermi gas",
  "rectified_doc_7.png": "The quantum square-well fluid: a thermodynamic geometric view",
  "rectified_doc_8.png": "Peacock’s Principle as a Conservative Strategy",
  "rectified_doc_9.png": "Task learning increases information redundancy of neural responses in macaque visual cortex",
  "rectified_doc_10.png": "Learning When to Look: On-Demand Keypoint-Video Fusion for Animal Behavior Analysis",
  "rectified_doc_11.png": "Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation",
  "rectified_doc_12.png": "Shadows and Polarization Images of a Four-dimensional Gauss-Bonnet Black Hole Irradiated by a Thick Accretion Disk",
  "rectified_doc_13.png": "A Dynamic Equilibrium Model for Automated Market Makers",
  "rectified_doc_14.png": "Nonconcave Portfolio Choice under Smooth Ambiguity",
  "rectified_doc_16.png": "Improved Constrained Generation by Bridging Pretrained Generative Models",
  "rectified_doc_17.png": "NATPS: Nonadiabatic Transition Path Sampling Using Time-Reversible MASH Dynamics",
  "rectified_doc_18.png": "Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models",
  "rectified_doc_19.png": "3D Dynamics of a Premagnetized Gas-puff Z-pinch implosion",
  "rectified_doc_20.png": "The Neural Compass: Probabilistic Relative Feature Fields for Robotic Search",
  "rectified_doc_21.png": "How Far Can Unsupervised RLVR Scale LLM Training?",
  "rectified_doc_22.png": "UNBOX: Unveiling Black-box visual models with Natural-language",
  "rectified_doc_24.png": "Summing to Uncertainty: On the Necessity of Additivity in Deriving the Born Rule",
  "rectified_doc_25.png": "Multi-sphere shape generator for DEM simulations of complex-shaped particles",
  "rectified_doc_26.png": "A Multilingual Human Annotated Corpus of Original and Easy-to-Read Texts",
  "rectified_doc_27.png": "Frequency of a Digit in the Representation of a Number and the Asymptotic Mean Value",
  "rectified_doc_28.png": "Polynomially many surfaces of fixed Euler characteristic in a hyperbolic 3-manifold",
  "rectified_doc_29.png": "A Curious Characterisation of Dedekind Domains",
  "rectified_doc_30.png": "English translation of Sophie Kowalevski’s ON THE PROBLEM OF THE ROTATION",
  "rectified_doc_31.png": "Non-invasive Growth Monitoring of Small Freshwater Fish in Home Aquariums",
  "rectified_doc_32.png": "Protein-Protein Interaction Sites Prediction Using Graph Convolutional Networks",
  "rectified_doc_33.png": "Hyperbolic elliptic parabolic disks and the representation of the group of isometries",
  "rectified_doc_34.png": "Deep learning for medical image segmentation: A review of recent developments",
  "rectified_doc_35.png": "A mathematical model for the dynamics of the COVID-19 pandemic including seasonal effects",
  "rectified_doc_37.png": "Computational modeling of protein-protein interactions: A survey of recent advances"
}

# Filter based on files actually present
docs_present = os.listdir(r"data/rectified_docs")
mapping_filtered = {k: v for k, v in mapping_all.items() if k in docs_present}

keys = sorted(list(mapping_filtered.keys()))
train_keys = keys[:25]
test_keys = keys[25:]

train_mapping = {k: mapping_filtered[k] for k in train_keys}
test_mapping = {k: mapping_filtered[k] for k in test_keys}

with open(r"data/rectified_train_mapping.json", "w") as f:
    json.dump(train_mapping, f, indent=4)

with open(r"data/rectified_test_mapping.json", "w") as f:
    json.dump(test_mapping, f, indent=4)

print(f"Created mappings: {len(train_mapping)} train, {len(test_mapping)} test documents.")
