import torch
from transformers import AutoTokenizer, AutoModel

from diffusion.schedule import Diffusion
from models.denoiser import Denoiser

device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer + embedding
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
embedder = bert.get_input_embeddings()

# diffusion + model
diffusion = Diffusion(device=device)
model = Denoiser(dim=embedder.embedding_dim).to(device)
model.load_state_dict(torch.load("denoiser.pt", map_location=device))
model.eval()

# parameters
batch_size = 1
seq_len = 8  # short on purpose
timesteps = diffusion.timesteps

# start from pure noise
x = torch.randn(
    batch_size,
    seq_len,
    embedder.embedding_dim,
    device=device
)

# reverse diffusion
for t in reversed(range(timesteps)):
    t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

    with torch.no_grad():
        pred_noise = model(x, t_batch)
        #pred_noise = pred_noise.clamp(-5.0, 5.0)

    alpha = diffusion.alphas[t]
    alpha_bar = diffusion.alpha_bar[t]

    x = (x - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()

# decode: embedding -> token logits
with torch.no_grad():
    logits = torch.matmul(x, embedder.weight.T)

tokens = torch.argmax(logits, dim=-1)
text = tokenizer.batch_decode(tokens, skip_special_tokens=True)

print("=== Generated Text ===")
print(text[0])
