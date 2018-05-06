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
[image4]: ./writeup_images/Validation_Class_Distribution.jpg             
"Validation_Class_Distribution"
[image5]: ./writeup_images/Test_Class_Distribution.jpg "Test_Class_Distribution"
[image6]: ./writeup_images/train_table.png "Train Table"
[image7]: ./writeup_images/org_hist.jpg "Original and Histogram Equalized"
[image8]: ./writeup_images/gray.jpg "Gray"
[image9]: ./writeup_images/flip2.jpg "Flip 2"
[image10]: ./writeup_images/flip1.jpg "Flip 1"
[image11]: ./writeup_images/rotate180.jpg "Rotate 180"
[image12]: ./writeup_images/rotate_bright.jpg "Rotate and Brightness"
[image13]: ./writeup_images/New_Training_Class_Distribution.jpg "Rotate and Brightness"


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

_10 random images from the train set_
![alt text][image1]
_1 image per class from the train set_
![alt text][image2]
_Train, Validation, Test distribution_
![alt text][image3]
![alt text][image4]
![alt text][image5]
_Table with classID distribution_
![alt text][image6]

### Design and Test a Model Architecture

#### 1. Description of preprocessing the image data. What techniques were chosen and why?

I tried to train the network with 3 different preprocessing approaches:
* Plain RGB images (1)
* Histogram equalized images (2)
* Grayscale images (3)

Here is an example of 15 traffic sign images before and after histogram equalization or grayscaling, respectively.

![alt text][image7]
![alt text][image8]

As a last step before feeding the data to the network, I normalized the image data because ...

#### if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to generate additional data because as seen in the data visualization section, the dataset is quite unbalanced. I tried to obtain a more balanced dataset to prevent the network from making biased predictions towards classes with plenty of images.

To add more data to the the data set, I used the following techniques:
* _(i)_ Flip images between 2 classes which are y-symmetric to each other (e.g. 36 <-> 37, ...)
* _(ii)_ Flip images of classes which are y-symmetric with themself (e.g. 30, ...)
* _(iii)_ Flip images (horizontal + vertical) of classes which are 180° rotation invariant
* _(iv)_ Augment images of classes which are underrepresented in the train set (random rotate between -30°/+30° and change brightness with gamma correction)

Here are examples of original images vs. augmented images for all of the above described 4 cases:
_(i)_
![alt text][image9]
_(ii)_
![alt text][image10]
_(iii)_
![alt text][image11]
_(iv)_
![alt text][image12]

The new train set distribution looks like this:
_Old_
![alt text][image3]
_New_
![alt text][image13]

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
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the final model, I used a batch size of 512, 320 epochs (with early stopping) and dropout (0.5) after the 3rd convolution layer, 1st fully connected layer and 2nd fully connected layer. Besides that, I used adam optimizer with a learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.979 
* test set accuracy of 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image14] ![alt text][image15] ![alt text][image16] 
![alt text][image17] ![alt text][image18]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


