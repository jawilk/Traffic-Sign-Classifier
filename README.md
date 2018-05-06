# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Augment the dataset
* Design, train and test a model architecture over an iterative process
* Use the model to make predictions on new images
* Analyze the top 5 softmax probabilities of the new images
* Visualize the output of different convolution layers

[//]: # (Image References)

[image1]: ./writeup_images/10_random.jpg "10 Random Images"
[image2]: ./writeup_images/1_per_class.jpg "1 per Cass"
[image3]: ./writeup_images/Training_Class_Distribution.jpg "Training Class Distribution"
[image4]: ./writeup_images/Validation_Class_Distribution.jpg "Validation_Class_Distribution"
[image5]: ./writeup_images/Test_Class_Distribution.jpg "Test_Class_Distribution"
[image6]: ./writeup_images/train_table.png "Train Table"
[image7]: ./writeup_images/org_hist.jpg "Original and Histogram Equalized"
[image8]: ./writeup_images/gray.jpg "Gray"
[image9]: ./writeup_images/flip2.jpg "Flip 2"
[image10]: ./writeup_images/flip1.jpg "Flip 1"
[image11]: ./writeup_images/rotate180.jpg "Rotate 180"
[image12]: ./writeup_images/rotate_bright.jpg "Rotate and Brightness"
[image13]: ./writeup_images/New_Training_Class_Distribution.jpg "Rotate and Brightness"
[image14]: ./writeup_images/lenet.png "LeNet"
[image15]: ./writeup_images/new_images.jpg "New Images"
[image16]: ./writeup_images/example1.png "Example 1"
[image17]: ./writeup_images/1activation1.png "1activation1"
[image18]: ./writeup_images/1activation2.png "1activation2"
[image19]: ./writeup_images/example2.png "Example 2"
[image20]: ./writeup_images/2activation1.png "2activation1"
[image21]: ./writeup_images/2activation2.png "2activation2"

---

### Writeup / README

### 1. Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The training set contains 34799 images
* The validation set contains 4410 images
* The test set contains 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. First 10 random images, then 1 example image of each class. Following 3 histogram bar charts showing how the train/validation/test data is distributed over the 43 classes and a table showing classID, name and frequency in the train set (ordered with frequency low -> high).

_10 random images from the train set_ <br />
![alt text][image1] <br />
_1 image per class from the train set_ <br />
![alt text][image2] <br />
_Training set distribution_ <br />
![alt text][image3] <br />
_Validation set distribution_ <br />
![alt text][image4] <br />
_Test set distribution_ <br />
![alt text][image5] <br />
_Table with classID distribution_ <br />
![alt text][image6] <br />

### Design and Test a Model Architecture

#### 1. Description of preprocessing the image data. What techniques were chosen and why?

I tried to train the network with 3 different preprocessing approaches:
* Plain RGB images (1)
* Histogram equalized images (2)
* Grayscale images (3)

Here is an example of 15 traffic sign images before and after histogram equalization or grayscaling, respectively.

![alt text][image7]
![alt text][image8]

As a last step before feeding the data to the network, I normalized the image data using Min-Max scaling.

#### if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to generate additional data because as seen in the data visualization section, the dataset is quite unbalanced. I tried to obtain a more balanced dataset to prevent the network from making biased predictions towards classes with plenty of images.

To add more data to the the data set, I used the following techniques:
* _(i)_ Flip images between 2 classes which are y-symmetric to each other (e.g. 36 <-> 37, ...)
* _(ii)_ Flip images of classes which are y-symmetric with themself (e.g. 30, ...)
* _(iii)_ Flip images (horizontal + vertical) of classes which are 180° rotation invariant
* _(iv)_ Augment images of classes which are underrepresented in the train set (random rotate between -30°/+30° and change brightness with gamma correction)

Here are examples of original images vs. augmented images for all of the above described 4 cases: <br />
_(i)_ <br />
![alt text][image9] <br />
_(ii)_ <br />
![alt text][image10] <br />
_(iii)_ <br />
![alt text][image11] <br />
_(iv)_ <br />
![alt text][image12] <br />

The new train set distribution looks like this: <br />
_Old_ <br />
![alt text][image3] <br />
_New_ <br />
![alt text][image13] <br />

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs = 10x10x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs = 5x5x64 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x64 	|
| RELU					|												|
| Dropout				|				0.5								|
| Fully connected		| outputs 240    									|
| Dropout				|				0.5								|
| Fully connected		| outputs 168        									|
| Dropout				|				0.5								|
| Fully connected		| outputs 43 (=classes)        									|
| Softmax				|        									|
 

#### 3. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**The approach:**

My first model I used was a plain LeNet model. <br />
![alt text][image14] <br />
Source: Yann LeCun

This model does a quite good job at classifying handwritten characters/digits. So I decided to give it a shot at classifying traffic sign images. It turns out to be a good model and I sticked to it (with small modifications) till the end.

After training with the plain model for 10 Epochs and a learning rate of 0.01 I got the following results:

_RGB images:_
* training set accuracy of 0.967
* validation set accuracy of 0.881

_Histogram equalized images:_
* training set accuracy of 0.958
* validation set accuracy of 0.903 

_Grayscale images:_
* training set accuracy of 0.970
* validation set accuracy of 0.883

In the following process I tried different filter sizes for both convolution layers and different numbers of neurons in the first and second fully connected layers. Besides that I tried adding a third convolution layer.
Further I trained the model on both augmented and not augmented data. Going further I adjusted the learning late and increased the number of epochs gradually. Additionally I used dropout/early stopping to prevent more overfitting.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.979 
* test set accuracy of 0.955

To train the final model, I used a batch size of 512, 320 epochs (with early stopping) and dropout (0.5) after the 3rd convolution layer, 1st fully connected layer and 2nd fully connected layer. Besides that, I used adam optimizer with a learning rate of 0.001.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 16 new traffic sign images I took in switzerland: <br />

![alt text][image15] <br />

These are almost euqivalent/similar to german ones. The model could have some issues due to downscaling (32x32). I added some images which might be hard to classify because they are occupied by stickers or bushes (e.g. images 12,15). Images 13 and 14 do not belong to the orginal training classes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield  									| 
|   Roundabout mandatory   			| Roundabout mandatory						|
| Yield					| Yield											|
| 	Keep right      		| Keep right				 				|
| Speed limit (30km/h)			|  Speed limit (30km/h)     							|
| Children crossing        | Bicycles crossing                   | 
|  No entry        | No entry                     |
| Ahead only          | Ahead only                   |
|  Road work         | Road work |             
| Turn right ahead      | Turn right ahead          |
| Priority road         | Priority road                   | 
|   Speed limit (60km/h)     | Speed limit (60km/h)                      |
| Yield          | Yield                      |
|   None          |  Speed limit (30km/h)                  |
| None      |  Dangerous curve to the right                    |
| Keep right      |  Keep right                    |





The model was able to correctly guess 13 of the 14 traffic signs (except 2 None classes), which gives an accuracy of **92.857%**. This compares favorably to the accuracy on the test set of **95.5%**.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions (top 5 softmax) on my final model is located at the end of the Ipython notebook.

---

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

_Example 1_ <br />
![alt text][image16] <br />
_1 Layer Activation_ <br />
![alt text][image17] <br />
_2 Layer Activation_ <br />
![alt text][image18] <br />
_Example 2_ <br />
![alt text][image19] <br />
_1 Layer Activation_ <br />
![alt text][image20] <br />
_2 Layer Activation_ <br />
![alt text][image21] <br />


