#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Creating the model
classifier = Sequential()
# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Flattening
classifier.add(Flatten())
# Full Neural Network
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting the images to CNN model
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('pics/train',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('pics/test',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
classifier.fit_generator(training_set,
steps_per_epoch = 15,
epochs = 10,
validation_data = test_set,
validation_steps = 30)

# Save the trained model
model_json = classifier.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model3.h5")
print("Saved model to disk")