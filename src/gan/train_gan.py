import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.gan.dcgan import Generator, Discriminator
from src.gan.gan_dataset import GANDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load only Glioma images
    dataset = GANDataset("data/raw/kaggle/training/glioma")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    epochs = 3
    z_dim = 100

    for epoch in range(epochs):
        for i, real_imgs in enumerate(loader):

            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------- Train Discriminator ----------------
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_imgs = G(z)

            D_real = D(real_imgs)
            D_fake = D(fake_imgs.detach())

            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---------------- Train Generator ----------------
            D_fake = D(fake_imgs)
            loss_G = criterion(D_fake, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

        # Save generated samples
        save_image(fake_imgs[:16], f"reports/figures/gan_epoch_{epoch+1}.png", normalize=True)

    torch.save(G.state_dict(), "saved_models/gan_generator.pth")
    print("✅ GAN Generator saved to saved_models/gan_generator.pth")


if __name__ == "__main__":
    main()
