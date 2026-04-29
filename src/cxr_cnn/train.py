
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cxr_cnn.config import Config

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    ''' set random seeds for reproducibility '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    ''' return GPU if available, otherwise CPU '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'using device: {device}')
    return device

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ) -> float:
    ''' run one training epoch, return mean loss'''
    
    model.train()
    total_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        # labels = labels.float().unsqueeze(1).to(device) # potential shape mismatch
        
        optimiser.zero_grad() # clear old grad
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # calculate loss
        loss.backward() # backpropagate gradients
        optimiser.step() # update weights
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ) -> tuple[float, float]:
    ''' evaluate model, return mean loss and accuracy '''
    
    model.eval() # model performance without updating weights
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # disable grad calculation, so eval is faster
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            # labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float() # logit -> sigmoid -> proba (probability > 0.5 -> pneumonia)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    return total_loss / len(loader), correct / total


def train(model: nn.Module, config: Config) -> None:
    ''' main training loop '''
    from cxr_cnn.dataset import get_dataloader
    
    set_seed(config.training.seed)
    device = get_device()
    model = model.to(device)
    
    training_loader = get_dataloader('train', config.data.image_size, config.data.batch_size, config.data.num_workers)
    val_loader = get_dataloader('val', config.data.image_size, config.data.batch_size, config.data.num_workers)
    
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    
    ## we need to avoid weight decay for BatchNorm weights and biases, so below is not optimal so below is not ideal.
    # optimiser = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-4)
    
    ## below should be proper approach (but need to change learning rate in config -> 1e-4)
    ## might be worth adding learning rate scheduler
    # decay_params = [p for n, p in model.named_parameters() if 'weight' in n and 'bn' not in n]
    # no_decay_params = [p for n, p in model.named_parameters() if 'weight' not in n or 'bn' in n]

    # optimiser = torch.optim.AdamW([
    #     {'params': decay_params, 'weight_decay': config.training.weight_decay},
    #     {'params': no_decay_params, 'weight_decay': 0.0},
    #     ], lr=config.training.learning_rate)
    
    for epoch in range(config.training.epochs):
        training_loss = train_one_epoch(model, training_loader, optimiser, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info(
            f'epoch {epoch + 1} / {config.training.epochs} | '
            f'training loss={training_loss:.4f} | '
            f'val_loss={val_loss:.4f} | '
            f'val_acc={val_acc:.4f}'
        )
            
    
    
    
