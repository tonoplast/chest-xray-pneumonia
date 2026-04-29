import logging

import medmnist
import numpy as np
import torch
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

def get_transforms(image_size: int, split: str) -> transforms.Compose:
    ''' return transforms for train or val/test split '''
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

def get_dataloader(split: str,
                   image_size: int,
                   batch_size: int,
                   num_workers: int,
                   ) -> DataLoader:
    ''' create dataloader for the given split '''
    assert split in ('train', 'val', 'test'), f'invalid split: {split}'
    
    dataset = PneumoniaMNIST(
        split=split,
        transform=get_transforms(image_size, split),
        download=True,
        size=image_size,
    )
    
    logger.info(f'{split} dataset loaded: {len(dataset)} samples')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers = num_workers,
    )