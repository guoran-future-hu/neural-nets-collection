import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import tarfile
import numpy as np
from tqdm import tqdm

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, reconstruction=False):
        self.root = root
        self.train = train
        self.transform = transform or transforms.ToTensor()
        self.reconstruction = reconstruction
        
        # Define the data directory
        self.data_dir = os.path.join(root, 'train' if train else 'val')
        
        self.images = []
        self.labels = []
        
        # Load data
        if train:
            # Process training data
            for class_idx, class_dir in enumerate(sorted(os.listdir(self.data_dir))):
                class_path = os.path.join(self.data_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(class_idx)
        else:
            # Process validation data
            val_annotations = os.path.join(root, 'val', 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                for line in f:
                    img_name, class_dir, *_ = line.strip().split('\t')
                    img_path = os.path.join(self.data_dir, 'images', img_name)
                    if os.path.exists(img_path):
                        self.images.append(img_path)
                        # Get class index from class directory name
                        class_idx = sorted(os.listdir(os.path.join(root, 'train'))).index(class_dir)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
        except:
            print(f"Error loading image: {img_path}")
            # Return a random different image
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform:
            img = self.transform(img)
            
        # For reconstruction tasks like VAE, return the image as both input and target
        if self.reconstruction:
            return img, img
        return img, label

def download_tiny_imagenet(root):
    """Download and extract Tiny ImageNet dataset"""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    if not os.path.exists(root):
        os.makedirs(root)
        
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    data_path = os.path.join(root, "tiny-imagenet-200")
    
    if os.path.exists(data_path):
        print("Tiny ImageNet dataset already downloaded and extracted.")
        return
    
    # Download if not exists
    print("Downloading Tiny ImageNet...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file, tqdm(
        desc=zip_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    # Extract if needed
    print("Extracting Tiny ImageNet...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)


def get_tiny_imagenet_datasets(root="./data", normalize=None, reconstruction=False, debug=False):
    """Get train and validation dataloaders for Tiny ImageNet"""
    
    download_tiny_imagenet(root)
    
    # Delete zip file after successful extraction
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # Define transforms
    transforms_list = [transforms.ToTensor()]
    if normalize == 'ImageNet':
        transforms_list.append(
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif normalize == 'MinMax':
        pass
    transform = transforms.Compose(transforms_list)
    
    # Create datasets
    train_dataset = TinyImageNet(
        root=os.path.join(root, "tiny-imagenet-200"),
        train=True,
        transform=transform,
        reconstruction=reconstruction
    )
    
    val_dataset = TinyImageNet(
        root=os.path.join(root, "tiny-imagenet-200"),
        train=False,
        transform=transform,
        reconstruction=reconstruction
    )
    
    if debug:
        # Only keep 1/10 of the data in debug mode
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            indices=range(0, len(train_dataset), 10)
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset,
            indices=range(0, len(val_dataset), 10)
        )
    
    return train_dataset, val_dataset

# Usage example:
if __name__ == "__main__":
    train_dataset, val_dataset = get_tiny_imagenet_datasets()
    
    # Print some statistics
    print(f"Number of training batches: {len(train_dataset)}")
    print(f"Number of validation batches: {len(val_dataset)}")
