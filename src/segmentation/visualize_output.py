import torch
import numpy as np
import matplotlib.pyplot as plt

from src.segmentation.unet import UNet


def main():
    # Load one sample image
    X = np.load("data/processed/kaggle/training/images.npy")

    # Convert RGB → grayscale
    img = np.mean(X[0], axis=2)

    # Convert to tensor (1,1,224,224)
    input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Load trained U-Net
    model = UNet()
    model.load_state_dict(torch.load("saved_models/unet_segmentation.pth", map_location="cpu"))
    model.eval()

    # Predict mask
    with torch.no_grad():
        pred_mask = model(input_tensor)

    pred_mask = pred_mask.squeeze().numpy()

    # Plot Results
    plt.figure(figsize=(12, 5))

    # Original MRI
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original MRI Image")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="jet")
    plt.title("Predicted Tumor Mask")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    plt.imshow(pred_mask, cmap="jet", alpha=0.5)
    plt.title("Tumor Mask Overlay")
    plt.axis("off")

    plt.tight_layout()

    # Save output for report
    plt.savefig("reports/figures/unet_segmentation_output.png", dpi=300)
    print("✅ Saved overlay output to reports/figures/unet_segmentation_output.png")

    plt.show()


if __name__ == "__main__":
    main()
