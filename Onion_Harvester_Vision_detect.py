import torch
from torchvision import transforms, models
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
args = parser.parse_args()

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('onion_model.pth'))
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = transform(img).unsqueeze(0)
pred = model(img)
label = 'Onion' if torch.argmax(pred) == 0 else 'Not Onion'
print(f"Detection: {label}")