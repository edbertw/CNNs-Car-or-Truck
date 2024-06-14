# Car or Truck?
## Overview
The code on this repository utilises Convolutional Neural Networks trained on an image based dataset to classify cars and trucks. The image dataset consists only of RGB colors with a defined size of (128,128)
## Methodology
1. Implemented data augmentation for better accuracy and more training data
2. Batch Normalization at each block of layers
3. Convolutional Layer for filtering images and defining of weights (kernel), as well as detection of image features using the non-linear RelU activation function and paddings
4. Maximum pooling layer to condense the image and obtain significant features; translation invariance
5. Flatten to convert 2D to 1D
6. Last layer has activation function sigmoid to output probability (0 to 1)
7. Optimization (Minimization) of the binary cross entropy with adam; binary accuracy as classification metrics
8. Train with 50 epochs
9. Plot line to determine accuracy through increasing epochs
## Results
Top accuracy : ```0.9048```
