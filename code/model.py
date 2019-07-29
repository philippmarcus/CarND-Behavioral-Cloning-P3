import csv
import cv2
import numpy as np
import copy
from sklearn.utils import shuffle

"""
Data generator and augmentation methods. The generator called
get_flipped_copies and get_color_inverted_copies, if augmentation mode
is activated.

The usage of color inverted copies is done to make the algorithm also work on streets that are brighter
than their environment.
"""

def get_flipped_copies(_X, _y):
        # Flip the image horizontally
        _X_flipped = list(map(lambda img: cv2.flip(img, 1), _X))
        _y_flipped = list(map(lambda angl: -angl, _y))
        return np.array(_X_flipped), _y_flipped

def get_color_inverted_copies(_X, _y):
        # Invert each channel of RGB image
        _X_inverted = list(map(lambda img: cv2.bitwise_not(img), _X))
        _y_inverted = copy.copy(_y)
        return np.array(_X_inverted), _y_inverted

def generator(samples, batch_size=32, augmentation=True):
        # add color inversed and flipped images later...
        batch_size = int(batch_size/3)
        num_samples = len(samples)

        # angle correction factors for center, left, right camera
        measurement_correction = {0:0., 1:0.2, 2:-0.2}

        while 1: # Loop forever so the generator never terminates\
                shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]

                        images = []
                        measurements = []
                        for batch_sample in batch_samples:
                                # For each parsed line, read in center, left, right image and angle
                                for i in range(3):
                                        source_path = batch_sample[i]
                                        lap = batch_sample[-1]
                                        filename = source_path.split("/")[-1]
                                        current_path = 'data/{}/IMG/'.format(lap) + filename
                                        img = cv2.imread(current_path)
                                        assert img is not None
                                        images.append(img)
                                        measurement = float(batch_sample[3]) + measurement_correction[i]
                                        measurements.append(measurement)
                        # Augmentation: Add color inverted and horizontally flipped versions
                        if augmentation:
                                X_mirrored, y_mirrored = get_flipped_copies(images, measurements)
                                images = np.concatenate((images, X_mirrored))
                                measurements = np.concatenate((measurements, y_mirrored))

                                X_clr_inv, y_clr_inv = get_color_inverted_copies(images, measurements)
                                images = np.concatenate((images, X_clr_inv))
                                measurements = np.concatenate((measurements, y_clr_inv))
                        images = np.array(images)
                        measurements = np.array(measurements)
                        yield shuffle(images, measurements)


"""
Read in the created sample files. A recording N needs to be placed
in a subfolder of data/lapN/

The generator for validation does not use data augmentation. Three data
sets were created. Two on track 1 and one on track 2.
"""
laps=["lap1", "lap2", "lap3"]
samples = []
for lap in laps:
        with open("data/{}/driving_log.csv".format(lap)) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                        line.append(lap)
                        samples.append(line)
shuffle(samples)
print("Collected {} raw samples".format(len(samples)))

# Set our batch size
batch_size=200

# Compile and train the model using the generator function
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Create the generators for training and test
train_generator = generator(train_samples, batch_size=batch_size, augmentation=True)
validation_generator = generator(validation_samples, batch_size=batch_size, augmentation=False)


"""
The Keras model based on the plaidml backend to run it on the GPU of a MacBook Pro.
"""
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Input, Activation, Dropout
from keras.layers.normalization import BatchNormalization

# Set to True to continue training on the stored model
CONTINUE_TRAINING = False

if CONTINUE_TRAINING:
        # Continue training on the imported model
        model = load_model("data/model.h5")
else:
        # Define the model and train from the scretch
        # Model based on: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        model = Sequential()

        #Preprocessing
        model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65,320,3)))

        # Network Model
        # Convolutional layers
        model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
        model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2), padding="valid"))
        model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
        model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid"))
        model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1), padding="valid"))
        model.add(Flatten())

        # Dense layers, all with dropout
        model.add(Dense(1064, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(100, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(50, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(Dense(1))

        # Use adam optimizer for adaptive learning rate
        model.compile(loss="mse", optimizer="adam")


from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

# Perform the training on the loaded or defined model
history_object = model.fit_generator(train_generator,\
        steps_per_epoch=np.ceil(len(train_samples)/batch_size),\
        validation_data=validation_generator,\
        validation_steps=np.ceil(len(validation_samples)/batch_size),\
        epochs=150,\
        verbose=1,
        callbacks = [es]
        )

model.save("data/model.h5")

"""
Plot the loss on training and validation data set
against the epochs.
"""
import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('learning_curve.png')