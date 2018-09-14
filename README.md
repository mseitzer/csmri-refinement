# Code for "Adversarial and Perceptual Refinement Compressed Sensing MRI Reconstruction"

This is the code release for the MICCAI 2018 paper Adversarial and Perceptual Refinement Compressed Sensing MRI Reconstruction by Maximilian Seitzer, Guang Yang, Jo Schlemper, Ozan Oktay, Tobias Würfl, Vincent Christlein, Tom Wong, Raad Mohiaddin, David Firmin, Jennifer Keegan, Daniel Rueckert and Andreas Maier.
You can find the paper on arXiv: [https://arxiv.org/abs/1806.11216].
In this work, we propose a way of training CNNs for MRI reconstruction that enhances perceptual quality of reconstructions while retaining higher PSNR than competing methods.

*As the dataset used in the paper is proprietary (we can unfortunately not release it), the code can ultimately serve only illustrative purposes.*
However, if you implement a loader for your own MRI dataset, you should be able to run the code with it. 
To this end, `data/reconstruction/scar_segmentation/scar_segmentation.py` may serve as a template for the loader's implementation.
The codebase is somewhat complex as it was implemented to be flexible and extensible. 

In `configs/`, three configuration files specifying the exact training parameters we used in the paper are given:
- `1-recnet.json`, for training the baseline reconstruction network using just MSE loss
- `2-refinement.json`, for training the refinement network using adversarial and perceptual losses
- `3-train-segmentation-unet.json`, for training the segmentation U-Net used to evaluate the semantic interpretability score
