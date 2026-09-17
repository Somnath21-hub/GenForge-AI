import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):

    def __init__(self, latent_size=32, num_classes=10):
        super().__init__()

        self.latent_size = latent_size
        self.num_classes = num_classes

        # --------------------------------
        # LABEL EMBEDDING
        # --------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            16
        )

        # --------------------------------
        # ENCODER
        # --------------------------------

        self.encoder = nn.Sequential(

            # 28 -> 14
            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 14 -> 7
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 7 -> 4
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Encoder output:
        # [128, 4, 4]
        #
        # 128 * 4 * 4 = 2048
        #
        # + 16 label embedding
        #
        # = 2064

        self.fc_mu = nn.Linear(
            2048 + 16,
            latent_size
        )

        self.fc_logvar = nn.Linear(
            2048 + 16,
            latent_size
        )

        # --------------------------------
        # DECODER
        # --------------------------------

        self.decoder_input = nn.Linear(
            latent_size + 16,
            128 * 4 * 4
        )

        self.decoder = nn.Sequential(

            # 4 -> 7
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 7 -> 14
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 14 -> 28
            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    # --------------------------------
    # ENCODE
    # --------------------------------

    def encode(self, image, labels):

        features = self.encoder(image)

        features = features.view(
            features.size(0),
            -1
        )

        label_features = self.label_embedding(
            labels
        )

        combined = torch.cat(
            [
                features,
                label_features
            ],
            dim=1
        )

        mu = self.fc_mu(combined)

        logvar = self.fc_logvar(combined)

        return mu, logvar

    # --------------------------------
    # REPARAMETERIZATION
    # --------------------------------

    def reparameterize(self, mu, logvar):

        std = torch.exp(
            0.5 * logvar
        )

        random_noise = torch.randn_like(std)

        z = mu + random_noise * std

        return z

    # --------------------------------
    # DECODE
    # --------------------------------

    def decode(self, z, labels):

        label_features = self.label_embedding(
            labels
        )

        combined = torch.cat(
            [
                z,
                label_features
            ],
            dim=1
        )

        x = self.decoder_input(
            combined
        )

        x = x.view(
            x.size(0),
            128,
            4,
            4
        )

        output = self.decoder(x)

        return output

    # --------------------------------
    # FORWARD
    # --------------------------------

    def forward(self, image, labels):

        mu, logvar = self.encode(
            image,
            labels
        )

        z = self.reparameterize(
            mu,
            logvar
        )

        output = self.decode(
            z,
            labels
        )

        return output, mu, logvar