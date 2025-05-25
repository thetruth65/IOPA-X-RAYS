# IOPA-X-RAYS ANALYSIS  
**By Mohit Sharma**

---

## 1. Problem Understanding

So here’s what I had to do: build an image processing pipeline to analyze `.dcm` medical imaging files (those DICOM X-ray scans). The goal was to make the workflow clean, insightful, and super easy to understand or scale later—basically, so that anyone could extend or build on it easily.

---

## 2. Folder Structure Description

Here’s how I organized everything:

- **Analysing_Scripts/**: Contains all my Jupyter notebooks for metadata analysis and static image processing (with outputs).
- **Comparison/**: A cool side-by-side comparison of ORIGINAL, STATIC, and ADAPTIVE enhanced images from my main pipeline.
- **Data/**: The raw dataset with 7 `.dcm` and 6 `.rvg` files.
- **Synthetic_Data/**: This is where I kept the noisy versions of the original images—used them to train the autoencoder.
- **Enhanced_Images/**: Output of the autoencoder, saved in both `128x128` and resized original dimensions.

---

## 3. Dependencies

Everything I used is in `requirements.txt`.

Core libraries:
- `numpy`, `pandas`, `scikit-image`, `pydicom` – for image enhancement, processing, and DICOM reading.

ML/Modeling:
- `scikit-learn`, `xgboost`, `optuna`, `torch` – used these for adaptive tuning and training the autoencoder (with CUDA support if available).

---

## 4. Basic Preprocessing

Step one: load and view the images. Then I applied:
- **Rescaling** – Resizes pixel values to a common range.
- **Windowing** – Focuses on specific intensity ranges, enhancing relevant features.
- **Normalization** – Scales intensity to [0, 1] to stabilize model input and output.

This brings every image to a similar scale so comparisons are fair.

I also calculated non-reference metrics like:
- Brightness – Measures overall lightness of an image.
- Sharpness – Indicates edge clarity and detail visibility.
- Contrast – Measures intensity difference between regions.
- Noise – Quantifies random pixel variations.

And yep, I visualized their intensity vs. frequency histograms. There was a clear imbalance in distributions (since real medical images are all over the place in terms of exposure).

---

## 5. Static Preprocessing

Next, I tried out classic image enhancement methods:

- **Histogram Equalization** – Redistributes intensity to enhance global contrast.
- **Unsharp Masking** – Highlights edges by subtracting a blurred copy from the original.
- **Bilateral Filtering** – Smooths while preserving edges using spatial and intensity differences.

I tested these one-by-one on each image, then applied them as a full pipeline. I’ve added comparisons (images + histograms) to show what changed.

---

## 6. Main Image Processing Pipeline

This is all wrapped up neatly in `image_processing.py`.  
It runs:
1. Basic Preprocessing  
2. Static Enhancement  
3. Adaptive Enhancement (explained next!)  
Then it logs and compares all results in a structured way.

---

## 7. Adaptive Preprocessing (Explained)

This is where things get a bit smarter.

Instead of hardcoding enhancement values, I let the system **predict the best parameters** for each image. Here’s how I did it:

1. I generated 50 artificial data samples based on the real images’ metric stats (mean, std, min, max).
2. Trained a **Random Forest Regressor** on the optimised results from Optuna to tune the target parameters of the techniques.  
   *(Random Forest helps capture nonlinear relationships; Optuna finds the best hyperparameters.)*
3. Predicted parameters for:
   - `clip_limit`, `tile_grid_size` (CLAHE) – controls local contrast enhancement in histograms.
   - `sigma_color`, `sigma_space` (Bilateral Filter) – balances spatial smoothing with edge protection.
   - `sharpen_weight` (Unsharp Mask) – adjusts how strongly edges are boosted.
4. Ran those enhancements with the predicted settings.

This made each image get its own optimal enhancement – sort of like a “personalized filter.”

---

## 8. Evaluation Metrics

I used both **non-reference** (no ground truth needed) and **reference** (compared to original) metrics.

### Non-Reference:
- Brightness – average pixel intensity.
- Sharpness – variance of Laplacian or similar edge estimator.
- Contrast – standard deviation of intensities.
- Noise – measured using difference between smoothed and original images.

### Reference:
- **PSNR (Peak Signal-to-Noise Ratio)** – Quantifies reconstruction quality (higher = better).
- **SSIM (Structural Similarity Index)** – Measures perceptual similarity based on structure, luminance, and contrast.

All metrics are saved as a CSV in the `Comparison/` folder.

---

## 9. Observations & Challenges

Alright, here’s the honest bit:

- **Static processing** actually performed better than **adaptive**, mainly because the dataset was so small.
- Brightness stayed decent (~0.5 across the board).
- Sharpness improved dramatically.
- Noise remained low (originals were already clean).
- Contrast improved slightly.

### Metric Summary:
- Static:
  - PSNR: **21.75**
  - SSIM: **0.83**
- Adaptive:
  - PSNR: **17.8**
  - SSIM: **0.71**

> If I had more data or did strong augmentation (lighting, rotation, noise), the adaptive model would’ve definitely improved.

---

## 10. Autoencoders

This part was fun. I trained a **PyTorch autoencoder** to remove noise from X-ray images.

Steps:
- I made noisy versions of the clean images (`Synthetic_Data/`) using Gaussian noise.  
  *(Gaussian noise simulates realistic sensor or transmission errors.)*
- Resized them to `128x128` for fast training.
- Built a basic encoder-decoder model (nothing fancy).  
  *(Encoder compresses image to latent features; decoder reconstructs it.)*
- Trained it and saved the denoised outputs to `Enhanced_Images/`.

### Thoughts:
- Visually better in many cases, especially on close inspection.
- Would get *way* better with:
  - **Patch-wise learning** – splits large images into smaller patches for training, helping the model focus on local structures and reduce memory load.
  - More data
  - Bigger architectures (like transformers or modern denoising nets; the attention mechanisms of theirs are mostly blindly reliable)

