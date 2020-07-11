from tensorflow.keras import Sequential
from keras.layers import Convolution2D
#convloution is done here

from keras.layers import MaxPooling2D
#for pooling

from keras.layers import Flatten
#helps in getting large feature vector

from keras.layers import Dense
#for adding fully connected layer in CNN

#intitalizing CNN
classifier = Sequential()

#1) adding convolution layer

#1st parameter - no. of filters, 2nd- no.of rows in each, 3rd- no. of columns in that
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation= 'relu'))
#32 filters,of 3*3 | input_shape=(size(dimension of 2d array, no of channels))

#2) pooling
#reduce size of feature map by 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation= 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#3) Flattening
classifier.add(Flatten())

#4) fully connected  layer
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

#compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

#preprocessing of images to avoid overfitting
#Image Augmentation process - to enrich the dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#target_size= dim of images
#class_mode = binary, whether the dependent variable is binary or not
training_set = train_datagen.flow_from_directory('C:/Users/ANKITA ADITYA/Desktop/Datasets/training_set',
                                                      target_size = (64,64),
                                                      batch_size = 32,
                                                      class_mode = 'binary')

test_datagen = test_datagen.flow_from_directory('C:/Users/ANKITA ADITYA/Desktop/Datasets/test_set',
                                                      target_size = (64,64),
                                                      batch_size = 32,
                                                      class_mode = 'binary')

#sample_per_epoch = total train images, nb_val_samples= no of images in test set
classifier.fit_generator(training_set,
                    steps_per_epoch=10000,
                    epochs=25,
                    validation_data='C:/Users/ANKITA ADITYA/Desktop/Datasets/test_set',
                    validation_steps=5000)
