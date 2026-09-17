import torch
from models.conditional_vae import ConditionalVAE


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


model = ConditionalVAE().to(device)


images = torch.randn(
    4, 1, 28, 28
).to(device)

labels = torch.tensor(
    [0, 1, 5, 9]
).to(device)


output, mu, logvar = model(
    images,
    labels
)


print("Input shape:", images.shape)
print("Labels shape:", labels.shape)
print("Output shape:", output.shape)
print("Mu shape:", mu.shape)
print("Logvar shape:", logvar.shape)