import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import cast


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DigitClassifier()
model.load_state_dict(torch.load("digitclassifier_torch.pth"))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x), 
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_image(image_path):
    pil_img = Image.open(image_path)
    tensor_img = cast(torch.Tensor, transform(pil_img)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_img)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


for i in range(7):
        img_path = f"{i}.jpg"
        digit = predict_image(img_path)
        print(f"Predicted Digit for {img_path}: {digit}")

