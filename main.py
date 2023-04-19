from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import torch.nn.functional as F
import os
import glob

app = FastAPI()

class PredictionResult(BaseModel):
    classification_results: List[str]
    score: List[float]

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]



@app.post("/predict", response_model=PredictionResponse)
async def check_model(image: UploadFile = File(None), uri: str=None):
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 80)
    # model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(8, 8))
    # model.classifier[0] = torch.nn.Linear(32768, 4096)
    data_dir = "./archive"
    train_dir = os.path.join(data_dir, "train")
    all_train_subdir = glob.glob(train_dir + "/*")

    classes=[os.path.basename(pp) for pp in all_train_subdir]
    classes = ['Hippopotamus', 'Sparrow', 'Magpie', 'Rhinoceros', 'Seahorse', 'Butterfly', 'Ladybug', 'Raccoon', 'Crab', 'Pig', 'Bull', 'Snail', 'Lynx', '.DS_Store', 'Turtle', 'Canary', 'Moths and butterflies', 'Fox', 'Cattle', 'Turkey', 'Scorpion', 'Goldfish', 'Giraffe', 'Bear', 'Penguin', 'Squid', 'Zebra', 'Brown bear', 'Leopard', 'Sheep', 'Hamster', 'Panda', 'Duck', 'Camel', 'Owl', 'Tiger', 'Whale', 'Crocodile', 'Eagle', 'Otter', 'Starfish', 'Goat', 'Jellyfish', 'Mule', 'Red panda', 'Raven', 'Mouse', 'Centipede', 'Lizard', 'Cheetah', 'Woodpecker', 'Sea lion', 'Shrimp', 'Polar bear', 'Parrot', 'Kangaroo', 'Worm', 'Caterpillar', 'Spider', 'Chicken', 'Monkey', 'Rabbit', 'Koala', 'Jaguar', 'Swan', 'Frog', 'Hedgehog', 'Sea turtle', 'Horse', 'Ostrich', 'Harbor seal', 'Fish', 'Squirrel', 'Deer', 'Lion', 'Goose', 'Shark', 'Tortoise', 'Snake', 'Elephant', 'Tick']

    model.load_state_dict(torch.load('./mobilenet_v2.pth'))

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    if image:
        img = Image.open(io.BytesIO(await image.read()))
    elif uri:
        if uri.startswith("http"):
            response = requests.get(uri)
            img = Image.open(io.BytesIO(response.content))
        else:
            img = Image.open(uri)

    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        score = torch.max(F.softmax(outputs, dim=1)).item()
        predicted = predicted.item()
        label = classes[predicted]
    return PredictionResponse(predictions=[PredictionResult(classification_results=[label], score=[score])])