import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, dim, n_layers=4):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.time_embed = nn.Embedding(1000, dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t).unsqueeze(1)
        x = x + t_emb
        return self.transformer(x)