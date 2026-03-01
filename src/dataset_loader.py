import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class DocumentDataset(Dataset):
    def __init__(self, data_dir, mapping_file, transform=None):
        """
        Args:
            data_dir (str): Directory containing the images.
            mapping_file (str): Path to the JSON file containing query-image mappings.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
            
        self.image_files = list(self.mapping.keys())
        self.queries = list(self.mapping.values())
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        query = self.queries[idx]
        
        return {"image": image, "query": query, "image_path": img_path}

def get_dataloader(data_dir, mapping_file, batch_size=4, shuffle=True, num_workers=0):
    dataset = DocumentDataset(data_dir, mapping_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
