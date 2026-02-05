import os
import torch
from torchvision.utils import save_image

from src.gan.dcgan import Generator


def main():
    os.makedirs("data/augmented/glioma_fake", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained Generator
    G = Generator().to(device)
    G.load_state_dict(torch.load("saved_models/gan_generator.pth", map_location=device))
    G.eval()

    z_dim = 100
    num_images = 100

    print("Generating synthetic MRI images...")

    for i in range(num_images):
        z = torch.randn(1, z_dim, 1, 1).to(device)
        fake_img = G(z)

        save_image(fake_img, f"data/augmented/glioma_fake/fake_{i}.png", normalize=True)

    print("✅ 100 Synthetic Glioma Images Saved in data/augmented/glioma_fake/")


if __name__ == "__main__":
    main()
