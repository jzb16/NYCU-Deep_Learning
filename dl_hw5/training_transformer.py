import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.device = args.device
        self.args = args
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.transformer.train()
        total_loss = 0
        for batch_idx, imgs in enumerate(train_loader):
            imgs = imgs.to(self.device)
            logits, target = self.model(imgs) # mask shape [10, 3, 64, 64]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(imgs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        print(f"====> Epoch: {epoch} Average loss: {total_loss / len(train_loader)}")
        print()


    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(self.device)
                logits, target = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"====> Validation loss: {val_loss:.4f}")
        print()
    

    def configure_optimizers(self):
        optimizer = Adam(self.model.transformer.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        return optimizer, scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=3, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:
    mask_schedule = MaskGit_CONFIGS['model_param']['gamma_type']
    print(mask_schedule)
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print('Training...')
        train_transformer.train_one_epoch(train_loader, epoch)
        print('Validating...')
        train_transformer.eval_one_epoch(val_loader, epoch)
        if epoch % 1 == 0:
            torch.save(train_transformer.model.state_dict(), f"transformer_checkpoints/ckpt_epoch_{mask_schedule}_{epoch}.pt")
        train_transformer.scheduler.step()