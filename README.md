# R2UNET-image-segmentaiton
This is an implementation of the retina blood vessel segmentation(medical image segmentation) using R2UNET architecture.

## DATABASE:
https://blogs.kingston.ac.uk/retinal/chasedb1/
The database contains 28 images of size 999 x 960 px of human retina and also their segmented images i.e desired output of the model
We will divide the images such that, training set will contain 20 images and their respective output images. The testing set will contain the remaining 8 images and their output images.

Example of input image:                                                 Corresponding desired output image:


























## MODEL ARCHITECTURE:
The architecture of R2Unet comprises of the following blocks:
Convolution Blocks 
This is the first layer and one of the main building blocks of a Convolutional Neural Networks (CNNs). 
They hold the raw pixel values of the training image as input.
This layer ensures the spatial relationship between pixels by learning image features using small squares of input data.

Recurrent Convolutional Block
Feature accumulation with recurrent convolutional layers ensures better feature representation for segmentation tasks. 
Recurrent network learns from neighbouring units which helps us to include context information of an image.






Encoding Block
Takes an input image and generates a high dimensional feature vector.
Aggregate features at multiple levels.

Decoding Block
Takes a high dimensional feature vector and generates a semantic segmentation mask. 
Decode features aggregated by encoder at multiple levels.

Skip Connections
The information from the initial layers is passed to deeper layers by matrix addition. 
The presence of the residual blocks prevents the loss of performance whenever the activations tend to vanish or explode by preserving the gradient.


