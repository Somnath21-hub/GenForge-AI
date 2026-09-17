import torch


def calculate_diversity(images):

    # Flatten each image
    images = images.view(images.size(0), -1)

    # Calculate distance between consecutive generated images
    distances = torch.norm(
        images[1:] - images[:-1],
        dim=1
    )

    # Average distance
    diversity_score = distances.mean().item()

    return diversity_score