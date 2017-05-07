# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/CenterLaneDriving.jpg "Center Lane Driving"

[image3]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/Recovery1.jpg "Recovery Image 1"
[image4]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/Recovery2.jpg "Recovery Image 2"
[image5]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/Recovery3.jpg "Recovery Image 3"
[image6]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/FlipExample_Original.png "Normal Image"
[image7]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/FlipExample_Flipped.png "Flipped Image"
[image8]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/LapVideoScreenshot.PNG "Lap Screenshot"
[image9]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ3/master/VideoScreenshot.PNG "Video Screenshot"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Files Submitted & Code Quality

#### 1.1 Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 1.2 Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 1.3 Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### 2. Model Architecture and Training Strategy

#### 2.1 An appropriate model architecture has been employed

The model architecture selected for the implementation of the Behavioral Cloning is the nVidia network. Initially the LeNet structure was tested but it was clear that the nVidia network obtained much better loss values in less epochs. The code can bee seen in **lines 100-114**

The model includes RELU layers to introduce nonlinearity (**code lines 103-107 excluding 104**), and the data is normalized in the model using a Keras lambda layer (code line 101). 

#### 2.2 Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (**model.py lines 107**). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 2.3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 2.4 Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Numerous amounts of data was gathered in order to correct the vehicle steering in order to stay on the designated 'safe' area.

For details about how I created the training data, see the next section. 

### 3.Model Architecture and Training Strategy

#### 3.1 Solution Design Approach

The overall strategy for deriving a model architecture was to decrease the mean squared error in the least amount of epochs.

My first step was to use a convolution neural network model similar to the LeNet structure I thought this model might be appropriate because it was used for the classification of Traffic signs in P2. Due to it being convolutional it was easy to implement the new shape of the images from this project (160x320x3).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. My firts model's mean squared error value was decreasing very slowly and many epochs were necessary. Also, the validation mean squared error was significatly higher than the same value of the test set which led me to believe that the model was overfitting the data.

I did not continue to use the LeNet structure and rebuilt my model according to the nVidia model. This model was significantly outperforming the Lenet model and satisfactory mean squared values were obtained after very few epochs. However, the same overfitting phenomenon was experienced even with the new model structure.

Therefore, I implemeneted a dropout function in order to decrease the overfitting characteristic. The attempt was successful.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. These areas were especially found where there was a change in the boundary of the track. This makes complete sense because there is not a lot of data available for these sections. In order to improve the performance of the model in these sections of track I gathered much more data in these sections, especially correcting manoeuvres.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at the highest speed it was tested of 25mph.

#### 3.2 Final Model Architecture

The final model architecture (**model.py lines 100-113**) characteristics are seen in the following table.

| Layer Number		| Description																						|
|:-----------		| :-------																							|
| **Layer 1**		| **Lambda Layer input=[160x320x3]**																|
| **Layer 2**		| **Cropping Layer output=[80x320x3]**																|
| **Layer 3**		| **Convolutional Layer, Stride=[5x5], subsample=[2x2], Activation=['RELU'] output Planes=[24]**	|
| **Layer 4**		| **Dropout Layer [25%]**																			|
| **Layer 5**		| **Convolutional Layer, Stride=[5x5], subsample=[2x2], Activation=['RELU'] output Planes=[36]**	|
| **Layer 6**		| **Convolutional Layer, Stride=[5x5], subsample=[2x2], Activation=['RELU'] output Planes=[48]**	|
| **Layer 7**		| **Convolutional Layer, Stride=[5x5], subsample=[2x2], Activation=['RELU'] output Planes=[64]**	|
| **Layer 8**		| **Flattening Layer**																				|
| **Layer 9**		| **Dense Layer Output=[100]**																		|
| **Layer 10**		| **Dense Layer Output=[50]**																		|
| **Layer 11**		| **Dense Layer Output=[10]**																		|
| **Layer 12**		| **Dense Layer Output=[1]**																		|
|||||

#### 3.3 Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving both clockwise and counterclockwise. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the center when getting close to the road boundaries. These images show what a recovery looks like starting from the right side of the track to the center of the track. Alos, the recovery was trained for the section where there is a gravel run off to the right. The rest of the track looks quite uniform therefore it is necessary to train a bit more in these sections in order to overcome the bias.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would create a more robust model. Doing so also increases the amount of data acquired (Doubling the data set). For example, here is an image that has then been flipped:

###### Original Image Steering Angle: 0.2358491
![alt text][image6]
###### Flipped Image Steering Angle: -0.2358491
![alt text][image7]


After the collection process, I had **11692** data points. The data was preprocessed by shuffling the array so that the order of the training data have no effect on the training of the model. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by the stagnation of the decrease rate of the training accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### 4. Video Results
Here are two videos showcasing the results form the Behavioral Cloning exercise:

#### 4.1 This video is a screen recording of the vehicle driving autonomously around the track
[![alt text][image8]](https://www.youtube.com/watch?v=1N7e5G_6VxQ&t=1s)

#### 4.2 This video was created using the video.py script
[![alt text][image9]](https://www.youtube.com/watch?v=xrjRfVf6xKU)
