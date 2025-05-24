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
- **Rescaling**  
- **Windowing**  
- **Normalization**  

This brings every image to a similar scale so comparisons are fair.

I also calculated non-reference metrics like:
- Brightness
- Sharpness
- Contrast
- Noise

And yep, I visualized their intensity vs. frequency histograms. There was a clear imbalance in distributions (since real medical images are all over the place in terms of exposure).

---

## 5. Static Preprocessing

Next, I tried out classic image enhancement methods:

- **Histogram Equalization** – stretches the intensity values to boost contrast.
- **Unsharp Masking** – adds clarity by enhancing edges.
- **Bilateral Filtering** – denoises while keeping edges crisp.

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
2. Trained a **Random Forest Regressor** using Optuna to tune it.
3. Predicted parameters for:
   - `clip_limit`, `tile_grid_size` (CLAHE)
   - `sigma_color`, `sigma_space` (Bilateral Filter)
   - `sharpen_weight` (Unsharp Mask)
4. Ran those enhancements with the predicted settings.

This made each image get its own optimal enhancement – sort of like a “personalized filter.”

---

## 8. Evaluation Metrics

I used both **non-reference** (no ground truth needed) and **reference** (compared to original) metrics.

### Non-Reference:
- Brightness
- Sharpness
- Contrast
- Noise

### Reference:
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better, less noise.
- **SSIM (Structural Similarity Index)**: Closer to 1 = more similar to original.

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
- Resized them to `128x128` for fast training.
- Built a basic encoder-decoder model (nothing fancy).
- Trained it and saved the denoised outputs to `Enhanced_Images/`.

### Thoughts:
- Visually better in many cases, especially on close inspection.
- Would get *way* better with:
  - Patch-wise learning
  - More data
  - Bigger architectures (like transformers or modern denoising nets)

---

## Final Thoughts

This project builds a solid, easy-to-read pipeline for preprocessing and enhancing medical X-rays. Whether you're trying static methods, tuning adaptively, or running ML models like autoencoders—everything's modular, scalable, and open to improvement.

If you're reading this to learn or build on it—go for it. Ping me if you wanna collaborate or brainstorm improvements!

