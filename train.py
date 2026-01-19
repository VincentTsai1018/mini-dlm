import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm

from data.dataset import TextDataset
from diffusion.schedule import Diffusion
from models.denoiser import Denoiser

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
dataset = TextDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Token embedding (frozen)
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
embedder = bert.get_input_embeddings()
embedder.requires_grad_(False)

# Diffusion
diffusion = Diffusion(device=device)

# Model
model = Denoiser(dim=embedder.embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step, tokens in enumerate(tqdm(loader)):
    tokens = tokens.to(device)

    with torch.no_grad():
        x0 = embedder(tokens)

    t = torch.randint(0, diffusion.timesteps, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise)

    pred_noise = model(xt, t)
    loss = torch.mean((pred_noise - noise) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

    if step >= 500:
        break