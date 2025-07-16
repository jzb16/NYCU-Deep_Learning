import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np

class LoadDataset(Dataset):
    def __init__(self, root, mode, _transforms=None):

        self.root = root
        self.mode = mode
        self.transforms = _transforms if _transforms else self.default_transforms()

        if mode == 'train':
            with open('train.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
            
        elif mode == 'test':
            with open('test.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.labels = self.json_data

        elif mode == 'new_test':
            with open('new_test.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.labels = self.json_data

        self.labels_one_hot = []
        with open('objects.json', 'r') as json_file:
            self.objects_dict = json.load(json_file)
            self.n_classes = len(self.objects_dict)

        for label in self.labels:
            label_one_hot = [0] * len(self.objects_dict)
            for text in label:
                label_one_hot[self.objects_dict[text]] = 1
            self.labels_one_hot.append(label_one_hot)
        
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            label_one_hot = torch.tensor(np.array(self.labels_one_hot[index]), dtype=torch.float32)
            return img, label_one_hot
        
        else:
            label_one_hot = torch.tensor(np.array(self.labels_one_hot[index]), dtype=torch.float32)
            return label_one_hot

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def create_dataloader(root, batch_size, num_workers, mode, _transforms=None):
    dataset = LoadDataset(root=root, mode=mode, _transforms=_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
    n_classes = dataset.n_classes # 24
    return dataloader, n_classes