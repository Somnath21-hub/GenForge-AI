import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import matplotlib.pyplot as plt

from models.conditional_vae import ConditionalVAE


# ========================================
# DEVICE
# ========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ========================================
# LOAD CVAE
# ========================================

model = ConditionalVAE(
    latent_size=32,
    num_classes=10
).to(device)

model.load_state_dict(
    torch.load(
        "models/conditional_vae.pth",
        map_location=device
    )
)

model.eval()

print("CVAE loaded successfully.")


# ========================================
# SAME LATENT VECTOR
# ========================================

z = torch.randn(
    1,
    32,
    device=device
)

images = []


# ========================================
# GENERATE SAME Z WITH DIFFERENT LABELS
# ========================================

with torch.no_grad():

    for class_id in range(10):

        label = torch.tensor(
            [class_id],
            dtype=torch.long,
            device=device
        )

        image = model.decode(
            z,
            label
        )

        images.append(
            image.squeeze().cpu()
        )


# ========================================
# CREATE GRID
# ========================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 4)
)


for class_id in range(10):

    axes.flat[class_id].imshow(
        images[class_id],
        cmap="gray"
    )

    axes.flat[class_id].set_title(
        f"Label {class_id}"
    )

    axes.flat[class_id].axis("off")


plt.tight_layout()


# ========================================
# SAVE IMAGE
# ========================================

os.makedirs(
    "generated",
    exist_ok=True
)

save_path = (
    "generated/"
    "label_conditioning.png"
)

plt.savefig(
    save_path,
    dpi=150
)

print(
    "\nSaved:",
    save_path
)

plt.show()