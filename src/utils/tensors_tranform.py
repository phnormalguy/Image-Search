def create_torch_tensor(data, labels):
    import torch
    data_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (N, C, H, W) format
    # Ensure labels are in the correct format
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor
    