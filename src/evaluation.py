import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LensPredictorNetwork
# from utils import extract_spatial_histogram

def predict_best_camera_parameter(image_path, model_path='saved_models/best_lens_predictor.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    model = LensPredictorNetwork(input_dim=160, output_dim=27).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image)
    
    # 3. Extract the lighting features
    light_vector = extract_spatial_histogram(img_tensor)
    
    # Add a batch dimension: [160] -> [1, 160]
    light_vector = light_vector.unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        logits = model(light_vector)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the highest probability
        confidence, predicted_class = torch.max(probabilities, 1)
        
    predicted_param = predicted_class.item() + 1 # Convert 0-26 back to 1-27
    
    print(f"File: {image_path}")
    print(f"Predicted Best Camera Setting: param_{predicted_param}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    
    return predicted_param

# Run a test
# predict_best_camera_parameter("my_test_image.jpg")