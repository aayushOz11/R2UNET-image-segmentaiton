# R2UNET-image-segmentaiton
This is an implementation of the retina blood vessel segmentation(medical image segmentation) using R2UNET architecture.

## DATABASE:
https://blogs.kingston.ac.uk/retinal/chasedb1/
The database contains 28 images of size 999 x 960 px of human retina and also their segmented images i.e desired output of the model
We will divide the images such that, training set will contain 20 images and their respective output images. The testing set will contain the remaining 8 images and their output images.

## Repository contents:
The app.py is the flask implementation of the R2UNET model where the model is deployed on the web.
The models foder must contain the h5 or the hdf5 format of the weigths of the model.
The main.py is the the code od the model if you want to implement locally in your PC.
Makes sure that you give correct file paths of training and testing sets that you will make when you download the database.

## Other information related to the model can be found in the report.pdf file in the project folder.
