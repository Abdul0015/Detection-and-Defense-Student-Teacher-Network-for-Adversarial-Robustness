import os
from torchvision import datasets
from PIL import Image

# Directory where images will be saved
output_dir = "cifar10_test_images"
os.makedirs(output_dir, exist_ok=True)

# Download CIFAR-10 test dataset
dataset = datasets.CIFAR10(root="./data", train=False, download=True)

# CIFAR-10 class names
classes = dataset.classes

# Save images
for idx, (img, label) in enumerate(dataset):
    class_name = classes[label]
    
    # Create class subfolder
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Save image
    img_path = os.path.join(class_dir, f"{idx}.png")
    img.save(img_path)

print("✅ CIFAR-10 test images saved successfully!")