from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torch
from torchvision import transforms
from feature_ext import ResNet18FeatureExtractor, create_data
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi import File, UploadFile
import shutil
import os 
import uuid
import base64
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, 'features/train_features.npy')
LABELS_PATH = os.path.join(BASE_DIR, 'features/train_labels.npy')
N_NEIGHBORS = 5

print(f"base_dir: {BASE_DIR}")


# --- Preprocessing ---
preprocess_fn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18FeatureExtractor()
model.to(device)
model.eval()

def extract_feature_vector(image_path, model, preprocess_fn, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    img = preprocess_fn(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(img).cpu().numpy().flatten()
    return feature

def visualize_top_k_similar_images(indices, train_data, train_labels, cifar10_classes, k=N_NEIGHBORS):
    plt.figure(figsize=(15, 3))
    for i in range(k):
        idx = indices[0][i]
        img = train_data[idx]
        label = train_labels[idx]
        plt.subplot(1, k, i + 1)
        plt.imshow(img)
        plt.title(f'Label: {cifar10_classes[label]}')
        plt.axis('off')
    plt.show()
    print(f"Top {k} similar images for the query image:")
    
# def save_neighbor_to_image(indices , train_data, train_labels, cifar10_classes, k=N_NEIGHBORS):
#     """Save top K similar images to disk."""
#     export_image_dir = "exported_images"
#     os.makedirs(export_image_dir, exist_ok=True)
#     for i in range(k):
#         idx = indices[0][i]
#         img = train_data[idx]
#         label = train_labels[idx]
#         img_path = os.path.join(export_image_dir, f"similar_{i+1}_label_{cifar10_classes[label]}.png")
#         img.save(img_path)
#         print(f"Saved similar image {i+1} with label {cifar10_classes[label]} to {img_path}")
#     return FileResponse(path=export_image_dir, media_type='application/zip', filename='exported_images.zip')
## load features
def main():
    # Load features and labels
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Fit NearestNeighbors
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='auto', metric='euclidean')
    nn.fit(features)

    # # Example query
    # query_image_path = 'image_test/bird1.jpeg'
    # query_feature = extract_feature_vector(query_image_path, model, preprocess_fn, device=device).reshape(1, -1)
    # distances, indices = nn.kneighbors(query_feature)
    # print("Top K similar image indices:", indices[0])
    # print("Distances:", distances[0])
    # print("Corresponding labels:", labels[indices[0]])

    # # Visualize
    # train_data, train_labels, _, _, cifar10_classes = create_data()
    # visualize_top_k_similar_images(indices, train_data, train_labels, cifar10_classes, k=N_NEIGHBORS)

# --- FastAPI App ---
@asynccontextmanager 
async def lifespan_events(app: FastAPI):
    """
    Context manager for managing application startup and shutdown events.
    The model loading logic is moved here to ensure it runs only once
    when the FastAPI application starts.
    """

    # Yield control to the application. Code after 'yield' runs on shutdown.
    yield
    print("Application shutdown: Cleaning up resources (if any)...")
    # You can add any cleanup logic here if needed, e.g., closing database connections
    # For a simple model serving, there might not be much to clean up.


# Initialize the FastAPI application with the lifespan event handler
app = FastAPI(lifespan=lifespan_events)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Search API"}

@app.post("/upload_search")
async def upload_and_search_image(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load features and fit NearestNeighbors
    features = np.load(FEATURES_PATH)
    train_data, train_labels, _, _, cifar10_classes = create_data()
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='auto', metric='euclidean')
    nn.fit(features)

    #Extract feature from uploaded image
    query_feature = extract_feature_vector(temp_path, model, preprocess_fn, device=device).reshape(1, -1)
    distances, indices = nn.kneighbors(query_feature)
    #Get top K similar images and their labels
    labels = train_labels[indices[0]]

    # Convert PIL images to base64 strings for frontend visualization

    image_base64_list = []
    for idx in indices[0]:
        img = train_data[idx]
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_base64_list.append(img_str)

    # Prepare response with images and labels
    # Clean up temp file
    os.remove(temp_path)
    

    return {
        "distances": distances[0].tolist(),
        "indices": indices[0].tolist(),
        "labels": labels.tolist(),
        "images": image_base64_list,
        
        
    }
    

@app.get("/download_features")
def download_features():
    file_path = os.path.join(BASE_DIR, "features/train_features.npy")
    return FileResponse(
        path=file_path,
        filename="train_features.npy",
        media_type="application/octet-stream"
    )
if __name__ == "__main__":
    main()