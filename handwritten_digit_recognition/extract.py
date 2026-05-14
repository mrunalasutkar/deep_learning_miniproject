from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("mnist_images", exist_ok=True)

# Load MNIST
dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Save images
for i in range(50):
    image, label = dataset[i]

    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")

    filename = f"mnist_images/{label}_{i}.png"

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

print("Images saved successfully!")