import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Dataset class for synthetic VSA scene data


class SyntheticVSADataset(Dataset):
    def __init__(self, image_dir, vector_dir, transform=None):
        self.image_dir = image_dir
        self.vector_dir = vector_dir
        self.transform = transform
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") and int(f.replace(".jpg", "")) >= 1000
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx].replace(".jpg", "")
        img_path = os.path.join(self.image_dir, f"{fname}.jpg")
        vec_path = os.path.join(self.vector_dir, f"{fname}.npy")

        img = Image.open(img_path).convert("L")
        vec = np.load(vec_path)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(vec, dtype=torch.float32)

# Neural network with two fully connected hidden layers


class VSAEncoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden1=1024, hidden2=2048, output_dim=10000):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Training setup


def train_model(image_dir, vector_dir, epochs=10, batch_size=32, learning_rate=1e-3, vsa_dim=10000):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    dataset = SyntheticVSADataset(image_dir, vector_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VSAEncoder(output_dim=vsa_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")

    return model


if __name__ == "__main__":
    model = train_model("synthetic_dataset_3/images",
                        "synthetic_dataset_3/vectors")
    torch.save(model.state_dict(), "vsa_encoder.pth")
