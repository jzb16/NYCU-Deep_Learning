import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
import torch.nn.functional as F
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))
        #self.transformer.load_state_dict(torch.load(load_ckpt_path), strict=True)

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_checkpoint(configs['VQ_CKPT_path'])
        #model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True)
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices.view(x.size(0), -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda t: t
        elif mode == "cosine":
            return lambda t: 1 - torch.cos(torch.pi * torch.tensor(t) * 0.5)
        elif mode == "square":
            return lambda t: (t ** 2)
        else:
            raise NotImplementedError("Mask function mode not supported.")

##TODO2 step1-3:            
    def forward(self, x):
        z, z_indices = self.encode_to_z(x) # z.shape [10, 3, 64, 64]
        z_indices = z_indices.view(x.size(0), -1) # z indices [10, 256]
        batch_size, num_tokens = z_indices.size()

        # Create a mask
        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices

        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=z_indices.device) # mask [10, 256]
        mask.scatter_(dim=1, index=sample, value=True)
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)

        # ~mask: False masked
        a_indices = mask * z_indices + (~mask) * masked_indices
        logits = self.transformer(a_indices)
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio):
        a_indices = z_indices.clone()
        a_indices[mask] = self.mask_token_id
        logits = self.transformer(a_indices)
        probs = F.softmax(logits, dim=-1)

        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g

        # update confidence
        confidence = torch.where(mask, confidence, torch.full_like(confidence, -float('inf')))

        _, sorted_indices = torch.sort(confidence, descending=True)
        mask_fill_count = math.ceil((1 - self.gamma(ratio)) * mask_num)

        new_mask = mask.clone()
        new_mask.fill_(False)
        new_mask.scatter_(dim=1, index=sorted_indices[:, :mask_fill_count], value=True)

        final_a_indices = new_mask * z_indices_predict + (~new_mask) * z_indices

        return final_a_indices, new_mask
        
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}