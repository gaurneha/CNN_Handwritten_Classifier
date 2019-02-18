
"""
Created on Fri Jan 11 19:00:33 2019

@author: neha
"""
import matplotlib as plt
import keras
from keras.datasets import mnist
pixel_width=28
pixel_height=28
#depth=1
(features_train,lables_train),(features_test,labels_test)=mnist.load_data()

print(features_train.shape)
features_train=features_train.reshape(features_train.shape[0],pixel_width,pixel_height,1)
#print(features_train.shape)
features_test=features_test.reshape(features_test.shape[0],pixel_width,pixel_height,1)
input_shape=(pixel_height,pixel_width,1)
#convert data into type float32
features_test=features_test.astype('float32')
features_train=features_train.astype('float32')
#each image is coming on height of 28,width of 28 and depth of 1
#since each image has black background with white and gey coding so color white is 255 and black is 0 and everything around 255 is grey
features_train/=255
features_test /=255
#print(features_train[0])
print(lables_train[5])

#below line  create a matrix and convert it into a binary matrix which place 0 in blank places and 1 in the value like for 2
#10 represents no of categorial values
lables_train=keras.utils.to_categorical(lables_train,10)
labels_test=keras.utils.to_categorical(labels_test,10)
print(lables_train[5])
# now data is ready to make prediction
#######################################
####################################

#Building Model using Sequential
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,Dropout

#create instance of Sequential
model=Sequential()
#to add a layer of complexity to machine learning model
#relu---- used to turn all negative values to '0'
#input_shape=tell us how how large the image we are dealing with
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
print("Post Conv2D: " ,model.output_shape)
#### pooling of model
model.add(MaxPooling2D(pool_size=(2,2)))
print("Post MaxPool: " ,model.output_shape)

#to prevent overfitting we passout Dropout layer
model.add(Dropout(0.25))

print("Post Dropout: " ,model.output_shape)

###add flatten layer to flat data into one flat pixel 13*13*32=5408
model.add(Flatten())
print("Post Flatten: " ,model.output_shape)
model.add(Dense(128,activation='relu'))
print("Post Denseof 128: " ,model.output_shape)

model.add(Dense(10,activation='relu'))
print("Post Dense: " ,model.output_shape)


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
#epoch is no of iterations

model.fit(features_train,lables_train,batch_size=32,epochs=3,verbose=1,validation_data=(features_test,labels_test))
score=model.evaluate(features_test,labels_test,verbose=0)

#
#model.save('handwritting_model.h5')
#
##coreml liberary is used in IOS to make prediction while wriiting something on it
#import coremltools
#coreml_model=coremltools.converters.keras.convert(model,input_names=['image'],image_input_names='image')
#
#coreml_model.author='neha'
#coreml_model.licence='abc'
#coreml_model.short_description='This model predict hand written image between 1-9'
#coreml_model.input_description['image']= 'A 28*28 pixel grayscale image.'
#coreml_model.output_description['output1']='A multiarray where the greatest float value(0-1) is the digit.'
#
##nmlmodel is an extension for coreml model
#coreml_model.save('handwritting_coreml_model.mlmodel')

#####################

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from numpy import array


#classifier = load_model('handwritting_model.h5')


test_image = image.load_img('/home/AI/CNN_ML/handwritting_images/6.png',target_size=(28,28))
test_image = image.img_to_array(test_image)
arr = array(test_image)

#print(test_image)
test_image = np.expand_dims(test_image, axis = 1)
test_image = test_image.reshape((-1, 28, 28,1))
yFit = model.predict(test_image, batch_size=10, verbose=1)

print(yFit)

#print(test_image)
#result = classifier.predict(test_image)
index = np.where(yFit == 1)[0]
number = (index - 1).item()
print("predicted number for my image: ", number)




