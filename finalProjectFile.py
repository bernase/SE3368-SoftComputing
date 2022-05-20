import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from keras_preprocessing.image import ImageDataGenerator

train_data_dir = 'Training'
validation_data_dir = 'Validation'
failure_train_dir = train_data_dir + '/Failure'
success_train_dir = train_data_dir + '/Success'
validation_failure_dir = validation_data_dir + '/Failure'
validation_success_dir = validation_data_dir + '/Success'


nb_train_samples = []
nb_validation_samples = []


nb_success = []
nb_failure = []

nb_train_labels = []
nb_valid_labels = []


nb_data_index = 0
nb_label_index = 1



jpg_files = os.listdir(failure_train_dir)
image = []
for i in jpg_files:
    image = cv2.imread(failure_train_dir + "/" + str(i))
    print(image)
    #nb_failure.append([image, 0])
    #nb_train_samples.append([image, 0])
    nb_train_samples.append(list(image))
    nb_train_labels.append(0)




jpg_files = os.listdir(success_train_dir)
image = []
for i in jpg_files:
    image = cv2.imread(success_train_dir + "/" + str(i))
    #nb_success.append(np.array([image, 1]))
    #nb_train_samples.append(np.array([image, 1]))
    nb_train_samples.append(list(image))
    nb_train_labels.append(1)


jpg_files = os.listdir(validation_failure_dir)
image = []
for i in jpg_files:
    image = cv2.imread(validation_failure_dir + "/" + str(i))
    #nb_failure.append(np.array([image, 0]))
    #nb_validation_samples.append(np.array([image, 0]))
    nb_validation_samples.append(list(image))
    nb_valid_labels.append(0)



jpg_files = os.listdir(validation_success_dir)
image = []
for i in jpg_files:
    image = cv2.imread(validation_success_dir + "/" + str(i))
    #nb_success.append(np.array([image, 1]))
    #nb_validation_samples.append(np.array([image, 1]))
    nb_validation_samples.append(list(image))
    nb_valid_labels.append(1)




#print(os.listdir(r'C:/Users/yigal/'))
img_path = r'MangoShirts/Success/MANGO- BEST SURFFING.JPG'
#image_sample = cv2.imread(img_path)
img = cv2.imread(img_path)  # USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
plt.imshow(img, cmap='gray')


#img = cv2.imread(
 #   "C:\\Users\\yigal\\OneDrive\\"
  #  "Masaüstü\\MangoShirts\\Success\\"
   # "MANGO- BEST SURFFING.JPG")



ksize = 15  #Use size that makes sense to the image and fetaure size. Large may not be good.
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5 #Large sigma on small features will fully miss the features.
theta = 1*np.pi/2  #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/4  #1/4 works best for angled.
gamma=0.9  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0.8  #Phase offset. I leave it to 0. (For hidden pic use 0.8)


kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

plt.imshow(kernel)

#img = cv2.imread('images/synthetic.jpg')
#img = cv2.imread('images/zebra.jpg')  #Image source wikipedia: https://en.wikipedia.org/wiki/Plains_zebra
img = cv2.imread(img_path) #USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
plt.imshow(img, cmap='gray')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

kernel_resized = cv2.resize(kernel, (400, 400))                    # Resize image


plt.imshow(kernel_resized)
plt.imshow(fimg, cmap='gray')

#cv2.imshow('Kernel', kernel_resized)
#cv2.imshow('Original Img.', img)
#cv2.imshow('Filtered', fimg)
#cv2.waitKey()
#cv2.destroyAllWindows()
#


epochs = 10
batch_size = 100
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(384, 512, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


opt = keras.optimizers.Adam(lr=0.5e-2)



model.compile(optimizer=opt, loss='binary_crossentropy' , metrics=['Accuracy'])


#train_datagen = ImageDataGenerator(
 #   rescale=1. / 255,
  #  shear_range=0.2,
   # zoom_range=0.2,
    #horizontal_flip=True)

#train_generator = train_datagen.flow_from_directory(
 #   train_data_dir,
  #  target_size = (img_width, img_height),
   # batch_size=batch_size,
    #class_mode='binary')

#test_datagen = ImageDataGenerator(rescale=1. / 255)

#validation_generator = test_datagen.flow_from_directory(
 #   validation_data_dir,
  #  target_size=(img_width, img_height),
   # batch_size=batch_size,
    #class_mode='binary')

#model.fit_generator(
 #   train_generator,
 #   steps_per_epoch=nb_train_samples // batch_size,
  #  epochs=epochs,
   # validation_data=validation_generator,
    #validation_steps=nb_validation_samples // batch_size)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=20)





train_x = np.array(nb_train_samples).reshape(-1, 355, 355, 1)
nb_train_labels_x = np.asarray(nb_train_labels)


hist = model.fit(train_x, nb_train_labels_x, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=1)
hist.history.keys()
model.summary()