# Traffic Sign Classifier Project

Overview
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. Specifically, you'll train a model to classify traffic signs from the German Traffic Sign Dataset.


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



[//]: # (Image References)

[image1]: ./images/data_hist.png "Data histogram"
[image2]: ./images/train_set.png "Train set"
[image3]: ./images/train_set_g.png "Train set grayscale"
[image4]: ./images/augmented_sample.png "Augmented dataset sample"
[image5]: ./img/img0_28.jpg "Traffic Sign 28"
[image6]: ./img/img1_9.jpg "Traffic Sign 9"
[image7]: ./img/img6_8.jpg "Traffic Sign 8"
[image8]: ./img/img7_14.jpg "Traffic Sign 14"
[image9]: ./img/img8_13.jpg "Traffic Sign 13"
[image10]: ./img/img9_34.jpg "Traffic Sign 34"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color information is not important in the recognition task. In addition, the dimension reduction allows to train the neural network faster.

Here is an example of a traffic sign image before grayscaling

![alt text][image2]

Here is an example of a traffic sign image after grayscaling

![alt text][image3]

As a last step, I normalized the image data because it will help the trainning process of the neural network.

I decided to generate additional data because some of the data sample classes have very few samples. Therefore, I try to generate samples based on the average samples. The objetive was to have a more balanced number of samples, in order to make the neural network more robust. 

To add more data to the the data set, I used the following techniques 


1) Compute the average number of samples (m) 
2) For each class that has fewer samples than the average (m), we perform the following:
	2.a) Generate k samples of the class, in order to have at least a number of samples equal to the average
	2.b) Each sample is generated using one of the following operation: translation, rotation, bright, blur and affine transformation. Each operation is choosed randomnly.

The objetive of this dataset was to make the neural network more robust to different inputs.

Here is an example of an original image and an augmented image:

![alt text][image4]

The difference between the original data set and the augmented data set is the following: the augmented dataset includes at least n samples, where n is equal to the average of samples of all the training data set classes.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    | 
| RELU					|												|
| Fully connected		| input 400, output 120                         | 
| RELU					|												|
| Dropout				| 50% keep      								|
| Fully connected		| input 120, output 84                          |
| RELU					|												|
| Dropout				| 50% keep      								|
| Fully connected		| input 84, output 43                           |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet model for the trainning, I made tests with dropout and since I get better results I choose to use them. The LeNet was trained with the AdamOptimizer, with a lerning rate of 0.001, 40 epochs, and a batch size of 100. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96%
* validation set accuracy of 93% 
* test set accuracy of 83%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the arquitecture proposed in the lectures.

* What were some problems with the initial architecture?
The accuracy of the initial arquitecture was below the requirement, however with a dropout of 50% the accuracy was better.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The arquitecture was modified to include dropouts of 50%

* Which parameters were tuned? How were they adjusted and why?
The parameters that were adjusted were learning rate, batch size, epochs, and drop out probability. Epochs and learning rate were adjusted to increase the accuracy. The drop out probability was used because the network accuracy was not enought and with it the accuraccy was better.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because is simmilar to sign traffic 11.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing     | Children crossing 							| 
| No vehicles      		| No vehicles   								| 
| Speed limit (120km/h) | Speed limit (120km/h)   						| 
| Stop Sign      		| Stop sign   									| 
| Yield					| Yield											|
| Turn left ahead     	| Turn left ahead 								|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 93%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a children crossing sign (probability of 0.98), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Children crossing   							| 
| .02     				| Right-of-way at the next intersection 		|


For the second image the model gives a No vehicles sign, which is correct

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        			    | No vehicles     							    | 

For the thirdh image the model gives a Speed limit (120km/h) sign, which is correct 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (120km/h)   						| 

For the fourth image the model gives a Stop Sign sign, which is correct 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop Sign    						            | 

For the fiveth image the model gives a Yield Sign sign, which is correct 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield    						            | 

For the sixth image the model gives a Turn left ahead Sign sign, which is correct 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Turn left ahead    						    | 



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

