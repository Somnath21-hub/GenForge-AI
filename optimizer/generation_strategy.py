import os
import sys
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.conditional_vae import ConditionalVAE


class GenerationStrategy:
    """
    GenForge Multi-Strategy Generation Engine.

    Supported Strategies:
    1. RANDOM_PRIOR:
       z ~ N(0, I) - Standard naive Gaussian prior generation.

    2. LATENT_MANIFOLD:
       Samples directly from encoded real class-specific latent representations
       with a controlled Gaussian perturbation: z = mu_real + scale * epsilon.

    3. ADAPTIVE_LATENT:
       Estimates empirical class-specific latent distribution parameters (mean mu_c, std sigma_c)
       from real encoded data and samples: z ~ N(mu_c, (scale * sigma_c)^2).
    """

    STRATEGY_RANDOM_PRIOR = "RANDOM_PRIOR"
    STRATEGY_LATENT_MANIFOLD = "LATENT_MANIFOLD"
    STRATEGY_ADAPTIVE_LATENT = "ADAPTIVE_LATENT"

    def __init__(
        self,
        model_path: str = "models/conditional_vae.pth",
        latent_size: int = 32,
        num_classes: int = 10,
        data_root: Optional[str] = None,
        device: Optional[torch.device] = None,
        lazy_latent_extraction: bool = False
    ):
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.model_path = os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path
        self.data_root = data_root or os.path.join(PROJECT_ROOT, "data")

        # --------------------------------
        # DEVICE SETUP
        # --------------------------------
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------------
        # CHECKPOINT VALIDATION & LOADING
        # --------------------------------
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[GenerationStrategy Error] Model checkpoint '{self.model_path}' not found! "
                f"Ensure the checkpoint exists before initializing GenerationStrategy. "
                f"GenForge will never silently fallback to an untrained model."
            )

        self.model = ConditionalVAE(
            latent_size=self.latent_size,
            num_classes=self.num_classes
        ).to(self.device)

        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ValueError(
                f"[GenerationStrategy Error] Failed to load checkpoint '{self.model_path}': {e}"
            )

        self.model.eval()

        # --------------------------------
        # CLASS LATENT POOLS & STATISTICS
        # --------------------------------
        self.class_latents: Dict[int, torch.Tensor] = {}
        self.class_means: Dict[int, torch.Tensor] = {}
        self.class_stds: Dict[int, torch.Tensor] = {}
        self._latents_extracted = False

        if not lazy_latent_extraction:
            self.extract_class_latents()

    def extract_class_latents(self, force_reload: bool = False):
        """
        Encodes the real training dataset through the CVAE encoder to build:
        - Class latent pools: M_c = {mu_i | y_i = c}
        - Class empirical mean: mu_c = mean(M_c)
        - Class empirical std: sigma_c = std(M_c)
        """
        if self._latents_extracted and not force_reload:
            return

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        pools = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                mu, _ = self.model.encode(images, labels)
                mu_cpu = mu.cpu()
                labels_cpu = labels.cpu()

                for c in range(self.num_classes):
                    mask = (labels_cpu == c)
                    if mask.any():
                        pools[c].append(mu_cpu[mask])

        for c in range(self.num_classes):
            if len(pools[c]) > 0:
                self.class_latents[c] = torch.cat(pools[c], dim=0)
                self.class_means[c] = self.class_latents[c].mean(dim=0).to(self.device)
                self.class_stds[c] = self.class_latents[c].std(dim=0).to(self.device)
                # Safeguard against zero std
                self.class_stds[c] = torch.clamp(self.class_stds[c], min=1e-4)
            else:
                self.class_latents[c] = torch.zeros((1, self.latent_size))
                self.class_means[c] = torch.zeros(self.latent_size, device=self.device)
                self.class_stds[c] = torch.ones(self.latent_size, device=self.device)

        self._latents_extracted = True

    # ====================================
    # 1. RANDOM PRIOR GENERATION
    # ====================================
    def random_prior(
        self,
        class_id: int,
        number_of_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples z ~ N(0, I) and decodes with requested class label.
        """
        labels = torch.full(
            (number_of_samples,),
            class_id,
            dtype=torch.long,
            device=self.device
        )

        noise = torch.randn(
            number_of_samples,
            self.latent_size,
            device=self.device
        )

        with torch.no_grad():
            images = self.model.decode(noise, labels)

        return images, labels

    # ====================================
    # 2. LATENT MANIFOLD RESAMPLING
    # ====================================
    def resample(
        self,
        class_id: int,
        number_of_samples: int,
        scale: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from real class-specific latent pool and perturbs by Gaussian noise:
        z = mu_pool + scale * N(0, I).
        """
        self.extract_class_latents()

        labels = torch.full(
            (number_of_samples,),
            class_id,
            dtype=torch.long,
            device=self.device
        )

        latent_pool = self.class_latents[class_id]
        if len(latent_pool) == 0:
            return self.random_prior(class_id, number_of_samples)

        indices = torch.randint(
            0,
            len(latent_pool),
            (number_of_samples,)
        )

        selected_mu = latent_pool[indices].to(self.device)
        noise = torch.randn(
            number_of_samples,
            self.latent_size,
            device=self.device
        ) * scale

        z = selected_mu + noise

        with torch.no_grad():
            images = self.model.decode(z, labels)

        return images, labels

    # ====================================
    # 3. ADAPTIVE LATENT GENERATION
    # ====================================
    def adapt_latent(
        self,
        class_id: int,
        number_of_samples: int,
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from estimated class-specific Gaussian distribution:
        z ~ N(mu_c, (scale * sigma_c)^2).
        """
        self.extract_class_latents()

        labels = torch.full(
            (number_of_samples,),
            class_id,
            dtype=torch.long,
            device=self.device
        )

        mu_c = self.class_means[class_id]
        sigma_c = self.class_stds[class_id]

        noise = torch.randn(
            number_of_samples,
            self.latent_size,
            device=self.device
        )

        z = mu_c + noise * (sigma_c * scale)

        with torch.no_grad():
            images = self.model.decode(z, labels)

        return images, labels

    # ====================================
    # UNIFIED GENERATE INTERFACE
    # ====================================
    def generate(
        self,
        class_id: int,
        number_of_samples: int,
        strategy: str = STRATEGY_LATENT_MANIFOLD,
        scale: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified generation method routing to requested strategy.
        """
        strat_upper = strategy.upper()
        if strat_upper in (self.STRATEGY_LATENT_MANIFOLD, "RESAMPLE", "LATENT_MANIFOLD"):
            return self.resample(class_id, number_of_samples, scale=scale)
        elif strat_upper in (self.STRATEGY_ADAPTIVE_LATENT, "ADAPT_LATENT", "ADAPTIVE_LATENT"):
            return self.adapt_latent(class_id, number_of_samples, scale=scale)
        elif strat_upper in (self.STRATEGY_RANDOM_PRIOR, "RANDOM_PRIOR", "RANDOM"):
            return self.random_prior(class_id, number_of_samples)
        else:
            print(f"[Warning] Unknown strategy '{strategy}', defaulting to LATENT_MANIFOLD.")
            return self.resample(class_id, number_of_samples, scale=scale)

    def generate_all(
        self,
        samples_per_class: int = 100,
        strategy: str = STRATEGY_LATENT_MANIFOLD,
        scale: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates balanced samples across all 10 classes.
        """
        all_images = []
        all_labels = []

        for c in range(self.num_classes):
            imgs, lbls = self.generate(
                class_id=c,
                number_of_samples=samples_per_class,
                strategy=strategy,
                scale=scale
            )
            all_images.append(imgs.cpu())
            all_labels.append(lbls.cpu())

        return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing GenerationStrategy Redesign")
    print("=" * 60)

    strat = GenerationStrategy(model_path="models/conditional_vae.pth")

    # 1. Test random prior
    imgs_rp, lbls_rp = strat.random_prior(class_id=5, number_of_samples=10)
    print("Random Prior shape:", imgs_rp.shape, "Labels:", lbls_rp[:5].tolist())

    # 2. Test latent manifold
    imgs_lm, lbls_lm = strat.resample(class_id=5, number_of_samples=10, scale=0.15)
    print("Latent Manifold shape:", imgs_lm.shape, "Labels:", lbls_lm[:5].tolist())

    # 3. Test adaptive latent
    imgs_al, lbls_al = strat.adapt_latent(class_id=5, number_of_samples=10, scale=1.0)
    print("Adaptive Latent shape:", imgs_al.shape, "Labels:", lbls_al[:5].tolist())

    print("\nGenerationStrategy tests passed successfully.")