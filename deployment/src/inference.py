
from fastapi import FastAPI, File, UploadFile
from typing import List
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import ResnetXInference

app = FastAPI()

# Load the trained model
model_path = 'saved_models/resnetX_cifar10.pth'
inference_model = ResnetXInference(num_classes=10)
inference_model.load_state_dict(torch.load(model_path))
inference_model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to match the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = inference_model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
