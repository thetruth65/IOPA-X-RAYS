import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pydicom
import cv2
from tqdm import tqdm

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, 'data')
synthetic_dir = os.path.join(root, 'synthetic_data')
enhanced_dir = os.path.join(root, 'enhanced_images')
model_path = os.path.join(root, 'model',  'autoencoder_best.pth')

# Create output directories if not exist
if not os.path.exists(synthetic_dir):
    os.makedirs(synthetic_dir)
if not os.path.exists(enhanced_dir):
    os.makedirs(enhanced_dir)

# Parameters
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
VAL_SPLIT = 0.2
NOISE_STD = 0.1

# Helper: Load and preprocess DICOM image with original size tracking
def load_dicom_image(path, return_original_size=False):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    original_shape = image.shape  # Store original dimensions
    
    intercept = float(ds.get('RescaleIntercept', 0))
    slope = float(ds.get('RescaleSlope', 1))
    image = image * slope + intercept
    
    # Windowing
    window_center = float(ds.get('WindowCenter', np.median(image)))
    window_width = float(ds.get('WindowWidth', np.ptp(image)))
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    image = np.clip(image, img_min, img_max)
    image = (image - img_min) / (img_max - img_min)
    image = np.clip(image, 0, 1)
    
    # Resize to IMG_SIZE x IMG_SIZE for training
    resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    if return_original_size:
        return resized_image, original_shape
    return resized_image

# Helper: Resize image back to original dimensions
def resize_to_original(image, original_shape):
    """Resize image back to original dimensions"""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3:
        image = image[0]  # Remove channel dimension if present
    return cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

# Helper: Save a single-channel torch tensor image as PNG using cv2
def save_tensor_image(tensor_img, save_path):
    if isinstance(tensor_img, torch.Tensor):
        img = tensor_img.detach().cpu().numpy()
    else:
        img = tensor_img
    if img.ndim == 3:
        img = img[0]  # (1, H, W) -> (H, W)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img)

# Custom Dataset for (noisy, clean) pairs with original size tracking
class DicomDenoiseDataset(Dataset):
    def __init__(self, dicom_paths, synthetic_dir, noise_std=0.1):
        self.dicom_paths = dicom_paths
        self.synthetic_dir = synthetic_dir
        self.noise_std = noise_std
        self.clean_images = []
        self.noisy_images = []
        self.original_shapes = []  # Store original dimensions
        self.prepare_data()

    def prepare_data(self):
        for path in self.dicom_paths:
            image, original_shape = load_dicom_image(path, return_original_size=True)
            self.original_shapes.append(original_shape)
            
            # Save clean image as tensor
            clean_tensor = torch.from_numpy(image).unsqueeze(0)
            self.clean_images.append(clean_tensor)
            
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            noisy = image + noise
            noisy = np.clip(noisy, 0, 1)
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)
            self.noisy_images.append(noisy_tensor)
            
            # Save noisy image as PNG (at 128x128 for reference)
            filename = os.path.splitext(os.path.basename(path))[0] + '_noisy.png'
            save_path = os.path.join(self.synthetic_dir, filename)
            save_tensor_image(noisy_tensor, save_path)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        return self.noisy_images[idx], self.clean_images[idx]
    
    def get_original_shape(self, idx):
        return self.original_shapes[idx]

# CNN Autoencoder (unchanged - works correctly for 128x128)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        # Bottleneck (fully connected)
        self.fc1 = nn.Linear(256*8*8, 512)
        self.fc2 = nn.Linear(512, 256*8*8)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), # 128x128
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.decoder(x)
        return x

# Load DICOM file paths
all_dicom_paths = []
dcm_files = glob.glob(os.path.join(data_dir, '*.dcm'))
for file in dcm_files:
    all_dicom_paths.append(file)
rvg_files = glob.glob(os.path.join(data_dir, '*.rvg'))
for file in rvg_files:
    all_dicom_paths.append(file)

print(f"Found {len(all_dicom_paths)} DICOM files")

# Prepare dataset
full_dataset = DicomDenoiseDataset(all_dicom_paths, synthetic_dir, noise_std=NOISE_STD)

total_size = len(full_dataset)
val_size = int(VAL_SPLIT * total_size)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {train_size}, Validation samples: {val_size}")

# Model, loss, optimizer
model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Training on device: {DEVICE}")

# Training loop
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for noisy, clean in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Training'):
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Validation'):
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            output = model(noisy)
            loss = criterion(output, clean)
            val_loss += loss.item() * noisy.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f'Best model saved at epoch {epoch+1} with val loss {val_loss:.6f}')

# Load best model
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

print("Generating enhanced images...")

# Enhance all noisy images and save at original resolution
for idx in range(len(full_dataset)):
    noisy, clean = full_dataset[idx]
    original_shape = full_dataset.get_original_shape(idx)
    
    noisy = noisy.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced = model(noisy)
    enhanced = enhanced.cpu().squeeze(0)
    
    # Resize enhanced image back to original dimensions
    enhanced_original_size = resize_to_original(enhanced, original_shape)
    
    # Save enhanced image at original resolution
    enhanced_filename = f'enhanced_{idx+1:03d}_original_size.png'
    enhanced_path = os.path.join(enhanced_dir, enhanced_filename)
    
    # Convert to uint8 and save
    enhanced_uint8 = np.clip(enhanced_original_size * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(enhanced_path, enhanced_uint8)
    
    # Also save the 128x128 version for comparison
    enhanced_filename_128 = f'enhanced_{idx+1:03d}_128x128.png'
    enhanced_path_128 = os.path.join(enhanced_dir, enhanced_filename_128)
    save_tensor_image(enhanced, enhanced_path_128)
    
    print(f"Saved enhanced image {idx+1}/{len(full_dataset)}: "
          f"Original size {original_shape} -> {enhanced_path}")

print('All enhanced images saved in', enhanced_dir)
print('All augmented (noisy) images saved in', synthetic_dir)
print('Best model saved as', model_path)


