import os
import shutil

src_dir = r"C:\Users\sange\.gemini\antigravity\brain\8da5e204-45d0-47da-b9ea-2dc38dc11cae"
dest_dir = r"c:\Users\sange\OneDrive\Desktop\DL_project\Visual-Document-Retrieval-Using-Vision-Transformers\data\rectified_docs"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

mapping = {
    "rectified_doc_1.png": "rectified_doc_1_1773182300721.png",
    "rectified_doc_2.png": "rectified_doc_2_1773182314910.png",
    "rectified_doc_3.png": "rectified_doc_3_1773182322574.png",
    "rectified_doc_4.png": "rectified_doc_4_1773182341707.png",
    "rectified_doc_5.png": "rectified_doc_5_1773182434911.png",
    "rectified_doc_6.png": "rectified_doc_6_1773182446997.png",
    "rectified_doc_7.png": "rectified_doc_7_1773182456780.png",
    "rectified_doc_8.png": "rectified_doc_8_1773182457508.png",
    "rectified_doc_9.png": "rectified_doc_9_1773182500735.png",
    "rectified_doc_10.png": "rectified_doc_10_1773182558678.png",
    "rectified_doc_11.png": "rectified_doc_11_1773182561121.png",
    "rectified_doc_12.png": "rectified_doc_12_1773182564454.png",
    "rectified_doc_13.png": "rectified_doc_13_success_1773182847631.png",
    "rectified_doc_14.png": "rectified_doc_14_success_1773182876342.png",
    "rectified_doc_16.png": "rectified_doc_16_1773182894518.png",
    "rectified_doc_17.png": "rectified_doc_17_retry_success_1773183413942.png",
    "rectified_doc_18.png": "rectified_doc_18_success_1773183337244.png",
    "rectified_doc_19.png": "rectified_doc_19_1773183441643.png",
    "rectified_doc_20.png": "rectified_doc_20_success_1773183486094.png",
    "rectified_doc_21.png": "rectified_doc_21_1773183544312.png",
    "rectified_doc_22.png": "rectified_doc_22_1773183558742.png",
    "rectified_doc_24.png": "rectified_doc_24_success_1773183622365.png",
    "rectified_doc_25.png": "rectified_doc_25_1773183628355.png",
    "rectified_doc_26.png": "rectified_doc_26_success_1773183639710.png",
    "rectified_doc_27.png": "rectified_doc_27_retry_1773183640825.png",
    "rectified_doc_28.png": "rectified_doc_28_retry_1773183641863.png",
    "rectified_doc_29.png": "rectified_doc_29_retry_1773183642861.png",
    "rectified_doc_30.png": "rectified_doc_30_1773183654626.png",
    "rectified_doc_31.png": "rectified_doc_31_retry_1773183655902.png",
    "rectified_doc_32.png": "rectified_doc_32_retry_2_1773183656760.png",
    "rectified_doc_33.png": "rectified_doc_33_retry_2_1773183658055.png",
    "rectified_doc_34.png": "rectified_doc_34_success_1773183095177.png",
    "rectified_doc_35.png": "rectified_doc_35_success_1773183225952.png",
    "rectified_doc_37.png": "rectified_doc_37_retry_2_1773183658990.png"
}

for dest, src in mapping.items():
    src_path = os.path.join(src_dir, src)
    dest_path = os.path.join(dest_dir, dest)
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        print(f"Copied {src} to {dest}")
    else:
        print(f"Warning: {src} not found")
