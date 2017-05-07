
# coding: utf-8

# In[3]:

### Importing the images
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time

dataDirectory = 'C:/Users/rvm/Documents/Ruan/Udacity/Self-Driving Car/TransferLearning/Behavioral_Cloning_P3/CarND-Behavioral-Cloning-P3/Data7/'

lines = []

with open(dataDirectory + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
steerLimit = 0.08 ### Setting a threshold for images to be kept otherwise the model would be biased to straight line driving


### Using the dataset csv file in order to import the necessary images.
for line in lines:
    measurement = float(line[3])
    steerFlag = random.randint(1,4)
    
    if np.abs(measurement)>steerLimit:
        
        sourcePath = line[0]
        filename = sourcePath.split('/')[-1]
        currentPath = dataDirectory + 'IMG/' + filename

        image = cv2.imread(currentPath)
        
        ### Preprocessing the images by firstly converting from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        ### Secondly flipping each image along with the sign of the steering angle value iin order to double the 
        ### amount of data acquired.
        image = np.fliplr(image)
        images.append(image)
        
        ### Here the steering angle measurement data is appended to an array.
        measurement = float(line[3])
        measurements.append(measurement)
        measurement = -measurement
        measurements.append(measurement)

### Here the data holders are created for training.
XTrain = np.array(images)
yTrain = np.array(measurements)

### The sequence of the data plays a very important role during training therefore the data is shuffled.
r = random.random()
random.shuffle(XTrain, lambda : r)
random.shuffle(yTrain, lambda : r)

### Data Details

print('The amount of images used for training is: ', len(XTrain))

    
### Visualise an Image        
index = random.randint(0,len(images))
imageTest = images[index]
cropImg = imageTest[65:145,0:320]
measureTest = measurements[index]
print('Steering Measure of image: ',measureTest)
#plt.imshow(cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB))
plt.imshow(imageTest)
plt.show()
print()
print('The Cropped image can be seen in the following figure')
#plt.imshow(cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB))
plt.imshow(cropImg)
plt.show()


# In[3]:

### Neural Network Architecture

startTime = time()

### Importing of necessary libraries
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

### Neural network stricture creation
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,15), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.35)) ### Dropout can decrease the overfitting effect
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 5, 5, subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(XTrain,yTrain, validation_split=0.30, shuffle='TRUE', nb_epoch=60, batch_size=128)

model.save('model.h5')


# In[4]:

print("Elapsed Time: ", (time()-startTime)/3600, 'Hours')

