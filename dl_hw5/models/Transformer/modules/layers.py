import torch.nn as nn
import torch
import math
import torch.nn.functional as F

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.d_k = dim // num_heads

        # Define linear layers for Q, K, V
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Scaling factor to prevent large values in the softmax denominator
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size = x.shape[0]
        
        # Linear projection split into q, k, v
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(self.scale.device)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(self.scale.device)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(self.scale.device)
    
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale #(MatMul between Q and K^T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum of V
        weighted_avg = torch.matmul(attn, v)
        weighted_avg = weighted_avg.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # Final linear layer
        weighted_avg = weighted_avg.to(self.device)
        output = self.out_linear(weighted_avg)
        
        return output

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    