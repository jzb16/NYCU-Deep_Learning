import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack
from tqdm import tqdm
#import imageio
import matplotlib.pyplot as plt
from math import log10
from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder
import torch.nn.functional as F

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.current_epoch = current_epoch
        self.total_epochs = args.num_epoch
        self.anneal_type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.beta = 1.0 if self.anneal_type.lower() == 'none' else 0.01


    def update(self, current_epoch):
        # TODO
        self.current_epoch = current_epoch
        if self.anneal_type.lower() == 'monotonic' or self.anneal_type.lower() == 'linear':
            print('Monotonic update')
            self.beta = min(1.0, ((self.current_epoch / self.cycle) * self.ratio))
        elif self.anneal_type.lower() == 'cyclical':
            print('Cyclical update')
            self.frange_cycle_linear(self.current_epoch, n_cycle=self.cycle, ratio=self.ratio)
        elif self.anneal_type.lower() == 'none':
            self.beta = 1.0
        else:
            raise ValueError("Unsupported annealing type. Choose 'linear', 'cyclical', or 'none'.")

        
    def get_beta(self):
        # TODO
        return self.beta
        
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=10, ratio=1): #ratio=1
        # TODO
        tmp = 0
        period = n_iter % n_cycle
        self.beta = min(1, (period / n_cycle) * ratio)
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        self.average_psnr = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, prev_img, label):
        # TODO  
        future_img_features = self.frame_transformation(img)
        future_label_features = self.label_transformation(label)

        if prev_img is not None:
            # Train
            z, mu, logvar = self.Gaussian_Predictor(future_img_features, future_label_features)
            prev_img_features = self.frame_transformation(prev_img)
            decoder_output = self.Decoder_Fusion(prev_img_features, future_label_features, z)
            output_img = self.Generator(decoder_output)
            return output_img, mu, logvar
        else:
            # Valid
            z = torch.randn(1, 12, 32, 64, device=img.device)
            decoder_output = self.Decoder_Fusion(future_img_features, future_label_features, z)
            output_img = self.Generator(decoder_output)
            return output_img
        
    
    def training_stage(self):
        losses, MSE_losses, KL_losses, psnr_list, beta_list, tfr_list = [], [], [], [], [], []
        Epochs = []
        for i in range(self.args.num_epoch):
            total_loss, total_mse_loss, total_kl_loss = 0, 0, 0
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse_loss, kl_loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += loss
                total_mse_loss += mse_loss
                total_kl_loss += kl_loss
                
                beta = self.kl_annealing.get_beta()
                self.tqdm_bar(f'train [TeacherForcing: {"ON" if adapt_TeacherForcing else "OFF"}, beta: {beta:.1f}]', pbar, loss, self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            # loss
            losses.append(total_loss / len(train_loader))
            MSE_losses.append(total_mse_loss / len(train_loader))
            KL_losses.append(total_kl_loss / len(train_loader))
            psnr_list.append(self.average_psnr)

            # tfr_beta
            tfr_list.append(self.tfr)
            beta_list.append(beta)

            self.eval()
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update(self.current_epoch)
            self.current_epoch += 1
            Epochs.append(i)

        self.plot_loss_curve(losses, MSE_losses, KL_losses, Epochs, self.args.kl_anneal_type)
        self.plot_tfr(tfr_list, beta_list, Epochs)
        self.plot_val_psnr(psnr_list, Epochs)

        print('Finish Training')
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss, self.scheduler.get_last_lr()[0])
        
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        self.train()
        self.optim.zero_grad()

        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)

        num_frames = img.shape[0]
        prev_output = img[0]

        mse_loss, kl_loss = 0, 0

        # Process each frame in the sequence
        for t in range(1, num_frames):
            current_label = label[t]
            current_img = img[t]

            if adapt_TeacherForcing:
                input_img = img[t-1]  # Use the actual previous frame
            else:
                input_img = prev_output  # Use the model-generated frame

            output, mu, logvar = self.forward(current_img, input_img, current_label)
            mse_loss += self.mse_criterion(output, current_img)
            kl_loss += kl_criterion(mu, logvar, current_img.shape[0])

            prev_output = output  # Update the previous output

        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optim.step()

        return loss, mse_loss, kl_loss
    
    def val_one_step(self, img, label):
        # TODO
        with torch.no_grad():
            img = img.permute(1, 0, 2, 3, 4)  # Change tensor into (seq, B, C, H, W)
            label = label.permute(1, 0, 2, 3, 4)

            psnrs = []
            num_frames = img.shape[0]
            frames_for_gif = []

            current_frame = img[0]
            frames_for_gif.append(current_frame.squeeze(0))

            mse_loss = 0

            # Generate and evaluate each subsequent frame
            for t in range(1, num_frames):
                current_label = label[t]
                next_frame = img[t]
                output = self.forward(current_frame, None, current_label)
                mse_loss += self.mse_criterion(output, next_frame)
                psnr = Generate_PSNR(output, img[t])
                psnrs.append(psnr.item())

                frames_for_gif.append(output.squeeze(0))
                current_frame = output

            self.average_psnr = sum(psnrs) / len(psnrs)
            self.make_gif(frames_for_gif, f'{args.save_root}/predicted_frames.gif')
            self.plot_psnr_per_frame(psnrs) 

        return mse_loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False) 
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.tfr_sde:
            self.tfr -= self.tfr_d_step
            self.tfr = max(self.tfr, 0)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(), #self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    
    def plot_tfr(self, tfr_list, beta_list, epoch_list):
        plt.plot(epoch_list, beta_list, label="beta")
        plt.plot(epoch_list, tfr_list, label="tfr_ratio")
        plt.xlabel("epochs")
        plt.title("TFR ratio and Beta")
        plt.legend()
        plt.savefig(f"{args.save_root}/tfr")
        plt.close()
    
    def plot_loss_curve(self, losses, MSE_losses, KL_losses, epoch_list, kl_anneal_type):
        losses = [loss.cpu().detach().numpy() for loss in losses]
        MSE_losses = [loss.cpu().detach().numpy() for loss in MSE_losses]
        KL_losses = [loss.cpu().detach().numpy() for loss in KL_losses]
        plt.plot(epoch_list, losses, label="total_loss")
        plt.plot(epoch_list, MSE_losses, label="mse_loss")
        plt.plot(epoch_list, KL_losses, label="kl_loss")
        plt.xlabel("epochs")
        plt.ylim(0, 1)
        plt.title(f"Loss Curve of {kl_anneal_type}")
        plt.legend()
        plt.savefig(f"{args.save_root}/loss_curve_{kl_anneal_type}")
        plt.close()
    
    def plot_psnr_per_frame(self, psnr_list):
        plt.figure(figsize=(10, 5))
        plt.plot(psnr_list, label='PSNR per Frame')
        plt.xlabel('Frame Index')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR per Frame During Validation')
        plt.legend()
        plt.savefig(f'{args.save_root}/psnr_per_frame.png')
        plt.close()

    def plot_val_psnr(self, val_psnr_list, epoch_list):
        plt.plot(epoch_list, val_psnr_list, label="val_psnr")
        plt.xlabel("epochs")
        plt.title(f"VAL PSNR")
        plt.legend()
        plt.savefig(f"{args.save_root}/VAL_PSNR")
        plt.close()

def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=3,     help="number of total epoch") #70
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="") # monotonic, cyclical, none
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="") #default=1
    
    args = parser.parse_args()
    
    main(args)
