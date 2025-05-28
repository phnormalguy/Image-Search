#%%
import sys
sys.path.append('src')
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torch
from torchvision import transforms
from feature_ext import ResNet18FeatureExtractor,create_data,visualize_images
import matplotlib.pyplot as plt



# Load features
features = np.load('features/train_features.npy')  
labels = np.load('features/train_labels.npy')  # Load labels if needed
print("Features shape:", features.shape)
print(f'type of features: {type(features)}')

# Fit NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
nn.fit(features)

# To find the 5 nearest neighbors of a query vector:
query = features[0].reshape(1, -1)
distances, indices = nn.kneighbors(query)
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Define preprocessing function for query images
preprocess_fn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # If you normalized your training images, add normalization here:
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your feature extractor model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18FeatureExtractor()
model.to(device)
model.eval()

# Function to extract feature vector from a query image
def extract_feature_vector(image_path, model, preprocess_fn, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    img = preprocess_fn(img).unsqueeze(0).to(device)  # Preprocess and add batch dimension
    with torch.no_grad():
        feature = model(img).cpu().numpy().flatten()
    return feature

# Example usage:
query_image_path = 'image_test/bird1.jpeg'
query_feature = extract_feature_vector(query_image_path, model, preprocess_fn, device=device).reshape(1, -1)
distances, indices = nn.kneighbors(query_feature)
print("Top K similar image indices:", indices[0])
print("Distances:", distances[0])

labels = np.load('features/train_labels.npy')  # Or load from your label source

print("Distances:", distances[0])
print("Corresponding labels:", labels[indices[0]])  

# Visualize the top K similar images
train_data, train_labels, test_data, test_labels, cifar10_classes = create_data()
def visualize_top_k_similar_images(indices, train_data, train_labels, cifar10_classes, k=5):
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
    
visualize_top_k_similar_images(indices, train_data, train_labels, cifar10_classes, k=5)   





# %%