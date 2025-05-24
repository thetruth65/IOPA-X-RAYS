import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import cv2
import optuna
from sklearn.ensemble import RandomForestRegressor
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import filters

# Set up main and output directories
main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
dist_dir = os.path.join(main_dir, 'comparison')
os.makedirs(dist_dir, exist_ok=True)

# Gather all DICOM and RVG files
image_files = []
image_files_dcm = glob.glob(os.path.join(data_dir, '*.dcm'))
for file in image_files_dcm:
    image_files.append(file)
image_files_rvg = glob.glob(os.path.join(data_dir, '*.rvg'))
for file in image_files_rvg:
    image_files.append(file)

# Helper functions
def load_dicom_image(file_path):
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array.astype(np.float32)
    intercept = float(ds.get('RescaleIntercept', 0))
    slope = float(ds.get('RescaleSlope', 1))
    image = image * slope + intercept
    window_center = float(ds.get('WindowCenter', np.median(image)))
    window_width = float(ds.get('WindowWidth', np.ptp(image)))
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    image = np.clip(image, img_min, img_max)
    image = (image - img_min) / (img_max - img_min)
    image = np.clip(image, 0, 1)
    return image, ds

def compute_sharpness(image):
    return cv2.Laplacian((image * 255).astype(np.uint8), cv2.CV_64F).var()

def compute_brightness(image):
    return np.mean(image)

def compute_contrast(image):
    return np.std(image)

def compute_noise(image):
    return np.std(image - filters.gaussian(image, sigma=2))

def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=1.0)

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0)

def static_preprocessing(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255
    gaussian = cv2.GaussianBlur(image_clahe, (5, 5), 0)
    sharpened = cv2.addWeighted(image_clahe, 1.5, gaussian, -0.5, 0)
    denoised = cv2.bilateralFilter((sharpened * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255
    return denoised

def apply_preprocessing(image, clip_limit=2.0, tile_grid_size=8, sharpen_weight=1.5, sigma_color=75, sigma_space=75):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    image_clahe = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255
    gaussian = cv2.GaussianBlur(image_clahe, (5, 5), 0)
    sharpened = cv2.addWeighted(image_clahe, sharpen_weight, gaussian, -0.5, 0)
    denoised = cv2.bilateralFilter((sharpened * 255).astype(np.uint8), 9, sigma_color, sigma_space).astype(np.float32) / 255
    return denoised

def optuna_objective(trial, image, reference):
    clip_limit = trial.suggest_float('clip_limit', 1.0, 5.0)
    tile_grid_size = trial.suggest_int('tile_grid_size', 4, 16)
    sharpen_weight = trial.suggest_float('sharpen_weight', 1.0, 2.5)
    sigma_color = trial.suggest_int('sigma_color', 50, 150)
    sigma_space = trial.suggest_int('sigma_space', 50, 150)
    processed = apply_preprocessing(image, clip_limit, tile_grid_size, sharpen_weight, sigma_color, sigma_space)
    return -compute_ssim(reference, processed)

def generate_artificial_data(num_samples=50):
    artificial_features = []
    artificial_targets = []
    for i in range(num_samples):
        mean = np.random.uniform(0.2, 0.8)
        std = np.random.uniform(0.05, 0.25)
        min_val = np.random.uniform(0.0, mean - 0.05)
        max_val = np.random.uniform(mean + 0.05, 1.0)
        features = [mean, std, min_val, max_val]
        clip_limit = np.random.uniform(1.0, 5.0)
        tile_grid_size = np.random.randint(4, 17)
        sharpen_weight = np.random.uniform(1.0, 2.5)
        sigma_color = np.random.randint(50, 151)
        sigma_space = np.random.randint(50, 151)
        targets = [clip_limit, tile_grid_size, sharpen_weight, sigma_color, sigma_space]
        artificial_features.append(features)
        artificial_targets.append(targets)
    return artificial_features, artificial_targets

# Optuna optimization and ML model training (15 trials)
optuna_trials = []
optuna_params = []
optuna_features = []
optuna_targets = []
for file_path in image_files[:3]:
    image, ds = load_dicom_image(file_path)
    study = optuna.create_study()
    study.optimize(lambda trial: optuna_objective(trial, image, image), n_trials=15, show_progress_bar=False)
    best_params = study.best_params
    optuna_trials.append(study)
    optuna_params.append(best_params)
    features = [np.mean(image), np.std(image), np.min(image), np.max(image)]
    optuna_features.append(features)
    optuna_targets.append([
        best_params['clip_limit'],
        best_params['tile_grid_size'],
        best_params['sharpen_weight'],
        best_params['sigma_color'],
        best_params['sigma_space']
    ])
if len(optuna_features) < 10:
    artificial_features, artificial_targets = generate_artificial_data(num_samples=50)
    for i in range(len(artificial_features)):
        optuna_features.append(artificial_features[i])
        optuna_targets.append(artificial_targets[i])
optuna_features = np.array(optuna_features)
optuna_targets = np.array(optuna_targets)
param_names = ['clip_limit', 'tile_grid_size', 'sharpen_weight', 'sigma_color', 'sigma_space']
regressors = []
for i in range(optuna_targets.shape[1]):
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(optuna_features, optuna_targets[:, i])
    regressors.append(rf)
def predict_params(image):
    features = np.array([np.mean(image), np.std(image), np.min(image), np.max(image)]).reshape(1, -1)
    preds = []
    for reg in regressors:
        preds.append(reg.predict(features)[0])
    preds[1] = int(np.clip(round(preds[1]), 4, 16))
    preds[3] = int(np.clip(round(preds[3]), 50, 150))
    preds[4] = int(np.clip(round(preds[4]), 50, 150))
    return preds
def adaptive_preprocessing(image):
    clip_limit, tile_grid_size, sharpen_weight, sigma_color, sigma_space = predict_params(image)
    return apply_preprocessing(image, clip_limit, tile_grid_size, sharpen_weight, sigma_color, sigma_space)

# Main processing loop
results = []
for file_path in image_files:
    image, ds = load_dicom_image(file_path)
    sharp = compute_sharpness(image)
    bright = compute_brightness(image)
    contr = compute_contrast(image)
    noise = compute_noise(image)
    static_img = static_preprocessing(image)
    adaptive_img = adaptive_preprocessing(image)
    metrics = {}
    metrics['filename'] = os.path.basename(file_path)
    metrics['sharpness_orig'] = sharp
    metrics['brightness_orig'] = bright
    metrics['contrast_orig'] = contr
    metrics['noise_orig'] = noise
    metrics['psnr_static'] = compute_psnr(image, static_img)
    metrics['ssim_static'] = compute_ssim(image, static_img)
    metrics['sharpness_static'] = compute_sharpness(static_img)
    metrics['brightness_static'] = compute_brightness(static_img)
    metrics['contrast_static'] = compute_contrast(static_img)
    metrics['noise_static'] = compute_noise(static_img)
    metrics['psnr_adaptive'] = compute_psnr(image, adaptive_img)
    metrics['ssim_adaptive'] = compute_ssim(image, adaptive_img)
    metrics['sharpness_adaptive'] = compute_sharpness(adaptive_img)
    metrics['brightness_adaptive'] = compute_brightness(adaptive_img)
    metrics['contrast_adaptive'] = compute_contrast(adaptive_img)
    metrics['noise_adaptive'] = compute_noise(adaptive_img)
    results.append(metrics)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(static_img, cmap='gray')
    axs[1].set_title('Static')
    axs[2].imshow(adaptive_img, cmap='gray')
    axs[2].set_title('Adaptive')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plot_path = os.path.join(dist_dir, os.path.basename(file_path) + '_comparison.png')
    plt.savefig(plot_path)
    plt.close()

# Save metrics to CSV
results_df = pd.DataFrame(results)
csv_path = os.path.join(dist_dir, 'image_metrics_comparison.csv')
results_df.to_csv(csv_path, index=False)
print('All results and plots saved in', dist_dir)



