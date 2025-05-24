**[IOPA-X-RAYS](https://github.com/thetruth65/IOPA-X-RAYS/tree/main)** **ANALYSIS**

**BY MOHIT SHARMA**

**1\. PROBLEM UNDERSTANDING**

- Design an image processing pipeline with evaluation metrics for insightful analysis and processing of medical imaging files(dicom format).
- Keep the workflow simple and in scalable format so that it’s easier for understanding and working further on it by others.

**2\. FOLDER STRUCTURE DESCRIPTION**

- Analysing Scripts folder contains jupyter notebooks and relevant output files describing the metadata and implementing the static processing techniques for analysis and comparison of images both visually and metric methods.
- Comparison folder contains comparison of images of ORIGINAL, STATIC and ADAPTIVE processed pipeline from the image_processing.py in the root folder.
- Data folder contains the exact provided data (7x .dcm and 6 x .rvg files).
- Synthetic_data folder contains a noisy copy of each image file from the data folder to feed it into our autoencoder for denoising in autoencoder.py python script in the root folder itself.
- Enhanced_images folder contains the imagers which are the output of autoencoder in 128x128 and original resized format.

**3\. DEPENDENCIES**

- Check the requirements.txt file for it.
- We have used basic numpy, scikit-image, pandas, pydicom and related libs for reading, manipulating and enhancing images and corresponding data.
- We have used scikit-learn, xgboost, optuna for models preprations and application in adaptive preprocessing along with torch with cuda help for autoencoders proof of concept.

**4\. BASIC PREPROCESSING**

- After reading the data from the dicom files and plotting it.
- Before that we perform the basic rescaling, windowing and image normalization in standard variable format for generalisation.
- Now, we calculate the non-reference metrics: brightness, sharpness, contrast and noise using the standard formulae.
- Then we plot their frequency vs intensity graph which we get to know is not evenly distribituted due to variation mentioned in the provided data instructions.

**5\. STATIC PREPROCESSING**

- Now to address the processing concepts, we apply the static techniques for more advancement in their quality.
- We use Histogram Equalization as it enhances image contrast by redistributing pixel intensities uniformly, improving visibility in under- or over-exposed regions.
- We Unsharp Masking as it sharpens images by subtracting a blurred version from the original, boosting edge contrast to enhance fine details.
- We use Bilateral filter as it smooths images while preserving edges by averaging pixels based on both spatial closeness and intensity similarity, making it ideal for denoising without blurring important features.
- And then we plot them in image format individually and compare them with the original provided ones.
- At end we apply them continuously in sequence format and then plot the image formats and histogram graphs for comparison of original and static_processed images.

**6.** **MAIN IMAGE PROCESSING**

- Now, in our main image_processing python script we apply all the techniques in sequence pipeline format.
- First, we do the basic preprocessing and apply the discussed static processing pipeline.
- Now we talk about the ADAPTIVE PREPROCESSING in the pipeline as in the end we compare all their image results and respective metrics.

**7\. ADAPTIVE PREPROCESSING(EXPLAINED)**

- Adaptive preprocessing refers to dynamically adjusting image enhancement techniques (like adaptive histogram equalization or thresholding) based on local image regions rather than globally, improving contrast or segmentation in varying lighting conditions. Key parameters include block size, clip limit (for contrast limiting), and neighbourhood radius (for local adaptation).
- The key to this technique is also to have as much data as possible to be able to try out different parameters on varying qualities of images for better generalisation over the tasks for which the images are to be used.
- First we generate some 50 artificial samples of the required parameters and their values to perform as target values and use the mean, std, min_val and max_val for as the features of the data so that we can train our ml algo on this artificial statistical numerical data giving a basic idea and example here in our pipeline.
- Here, we have used machine learning algorithm “Random Forest” combined with optuna for optimising the hyperparameters of Random Forest Regressor algorithm.
- Random Forest is a solid bagging technique providing randomness to data which helps in lowering the bias and variance.
- Now the techniques used in our static preprocessing pipeline that is unsharp masking and bilateral filter along with CLAHE(enhances local contrast by applying histogram equalization to small image tiles with a contrast limit to prevent noise amplification in homogeneous regions.); we use or optimise their parameters and clip their results accordingly using the ml algo and optimising techniques over their parameters of : clip_limit, tile_grid_size, sharpen_weight, sigma_color, sigma_space.
- Then we use the best parameters predicted in our techniques.
- At end we compare the results with original images , adaptive and static processed images of our pipeline.

**8\. EVALUATION METRICS**

- We use both Reference and Non reference methods of evaluation at almost everywhere after processing our images and even before processing them(for non-reference ones).
- NON- Referene Methods that are used are brightness, sharpness, contrast and noise. Which we have calculated using standerd basic formulae for checking the quality of images before and after processing as these are the most authentic without any dependence on ground truth images and all.
- Reference Methods that are used PSNR and SSIM tht is:
- PSNR – Peak Signal-to-Noise Ratio: A reference-based metric that compares the original and processed image based on pixel intensity differences. Measures how much noise or distortion has been introduced during processing. A higher **PSNR** (in decibels) indicates the processed image is closer to the original—**better quality, less distortion**.
- SSIM – Structural Similarity Index Measure: A reference-based perceptual metric that compares structural information like luminance, contrast, and texture between original and processed images. Captures **human visual perception** better than PSNR. **SSIM ranges from -1 to 1**, where **1 means identical**; higher SSIM implies **better structural fidelity and visual similarity** to the original.
- The csv file in our Comparison folder has all the details of the images eval metrics comparison for both reference and non-reference metrics for all original, static and adaptive processed images.

**9\. OBSERVATIONS FROM EVAL METRICS AND MENTIONED CHALLENGES ADDRESSED**

- We observe that the mean performance of the static processed images are better than that of adaptive processed ones over all the 13 images.
- Let’s discuss why that is: Because the adaptive processing techniques despite using advanced predictive techniques and algos for parameters, significantly depend on the data to learn and test on as they include a regress process of learning to be able to get the hold of generalised quality required for the image to be processed.
- Then we observe not much changes in brightness as ot is around 0.5 in all which is very decent for performing operations on in future.
- The sharpness has been greatly increased by both static and adaptive techniques which is good for getting clearer image view and perform further operations.
- We don’t observe much change in the noise of the images as they are originally very lest in 0.01 order which is good and difficult to denoise, so the denoising results are around of the same order to that of the original provided images.
- The contrast of the images has also slightly increased for both the pipelines with almost same increment in the every image.
- The avg PSNR of static is 21.75 and adaptive is 17.8 ; the reason being that of limited data but also we should rely more on SSIM for perception; so avg SSIM of static is 0.83 and adaptive is 0.71; which is decent comparsion as we look throughout their comparative iamegs we also some noisy data in the images of adaptive processed pipeline quite strongly due to less amount of data.
- We should have used data augmentation with varies gausian noise, lighting , sharpness , rotations, and all ; to get variations of data in huge numbers and then test on them.

**10\. AUTOENCODERS**

- Neural networks that learn efficient image representations by compressing (encoding) and reconstructing (decoding) data, making them ideal for **denoising, compression, and anomaly detection** under **semi-supervised or unsupervised** conditions in image preprocessing pipelines.
- Here we have made a synthetic folder files which are made by adding gaussian noise to the original provided images(1 copy each) and used them for train and validation.
- Ideally, yes we should have applied intensive data augmentation but due to computational time and space ; and just for application point of view , we are going with bare basic.
- We resize the images to 128 for computational ease and also this helps us to deal with noise till smaller resolution.
- Now ideally what we can do is divide the whole image into small pixel patches and then put them in autoencoder and then recombine them and resize them for performing outstanding results.
- But here we r just using our custom made autoencoder using pytorch , for denoising our resized images and returning the enhanced images both in 128 and resized format in our enhanced_images directory.
- If we observe them closely side by side in zoomed manner we do see the denoising of them however this can be significantly improved by using the above mentioned technique, data augmentation and state of the art autoencoders and transformers as their complex attention mechanisms are clear champions in these situations. However, they do increase computational time and space but perform really well.
