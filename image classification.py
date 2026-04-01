import torch
from torchvision import models, transforms
from PIL import Image
# Removed requests and io as we will read from local file

model = models.resnet18(pretrained=True)
model.eval()

# Load the image from the locally saved file
img = Image.open("dog.jpg")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = transform(img) # Apply transform to the PIL Image
img = img.unsqueeze(0)   # add batch dimension

with torch.no_grad():
    output = model(img)

_, predicted = torch.max(output, 1)

print("Predicted Class Index: ", predicted.item())