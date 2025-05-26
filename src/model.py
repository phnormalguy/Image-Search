
#%%
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
import cv2
from utils.unpickle import unpickle
from utils.tensors_tranform import create_torch_tensor
from utils.data_preprocessing import normalize_images
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



import os 
 
import matplotlib.pyplot as plt


def extract_features(model, data_tensor, batch_size=128, device='cpu'):
    model.eval()
    features_list = []
    loader = DataLoader(data_tensor, batch_size=batch_size)
    total = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            # Resize batch to 224x224 for ResNet-18
            batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
            feats = model(batch)
            features_list.append(feats.cpu())
            percent = (i + 1) / total * 100
            print(f"\rExtracting features: {percent:.1f}%", end="")
    print()  # for newline after progress
    return torch.cat(features_list, dim=0)

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        # Check if the pretrained model is available
        if os.path.exists("pretrain/resnet18-weights.pth"):
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.load_state_dict(torch.load("pretrain/resnet18-weights.pth"))
        else:
            self.resnet = models.resnet18(pretrained=True)
            os.makedirs("pretrain", exist_ok=True)
            torch.save(self.resnet.state_dict(), "pretrain/resnet18-weights.pth")
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Set the model to evaluation mode
        self.resnet.eval()
    
    def forward(self,x):
        # Pass the input through the ResNet-18 model
        x = self.resnet(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        return x

def visualize_images(images, labels, class_names, num_images=10):
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i])
            plt.title(class_names[labels[i]])
            plt.axis('off')
        plt.show()
        
        
        
if __name__ == "__main__":
    model = ResNet18FeatureExtractor(pretrained=True)
    
    train_data = []
    train_labels = []
    for batch in range(1,6):
        batch_file = fbatch_file = f"/home/june/Documents/GitHub/Image-Search/src/dataset/cifar-10-python/cifar-10-batches-py/data_batch_{batch}"
        batch_dict = unpickle(batch_file)
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])
    train_data = np.concatenate(train_data, axis=0)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)
    cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
    test_dict = unpickle("/home/june/Documents/GitHub/Image-Search/src/dataset/cifar-10-python/cifar-10-batches-py/test_batch")
    test_data = test_dict[b'data']
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_dict[b'labels'])
    
    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)


 ## Normalize the images
    train_data = normalize_images(train_data)
    test_data = normalize_images(test_data)
    
    print("After normalization, train data min:", train_data.min(), "max:", train_data.max())
    print("After normalization, test data min:", test_data.min(), "max:", test_data.max())
    
    

    # Convert to torch tensors
    train_data_tensor, train_labels_tensor = create_torch_tensor(train_data, train_labels)
    test_data_tensor, test_labels_tensor = create_torch_tensor(test_data, test_labels)
    
    print("Train data tensor shape:", train_data_tensor.shape)
    print("Train labels tensor shape:", train_labels_tensor.shape)
    print("Type of train data tensor:", type(train_data_tensor))
    print("Test data tensor shape:", test_data_tensor.shape)
    print("Test labels tensor shape:", test_labels_tensor.shape)
    print("Type of test data tensor:", type(test_data_tensor)) 

    
    print("After resizing, train data tensor shape:", sample_data.shape)
    print("After resizing, test data tensor shape:", sample_test_data.shape)
    
    ## Visualize some images
    visualize_images(train_data, train_labels, cifar10_classes, num_images=10)
    
    
    ## Extract features using ResNet-18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = extract_features(model, train_data_tensor, batch_size=128, device=device)
    print("All features shape:", features.shape)
    
    

    



        
    
        
# %%
