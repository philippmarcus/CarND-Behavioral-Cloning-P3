# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/model_final.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image81]: ./writeup/dangerous-situation-1.png "Dangerous Situation 1"
[image82]: ./writeup/dangerous-situation-2.png "Dangerous Situation 2"
[image83]: ./writeup/dangerous-situation-3.png "Dangerous Situation 3"
[image84]: ./writeup/dangerous-situation-4.png "Dangerous Situation 4"
[image11]: ./writeup/learning_curve.png "Learning Curve"
[image12]: ./writeup/track_1_original.png "Original Image Track 1"
[image13]: ./writeup/track_1_cropped.png "Cropped Image Track 1"
[image14]: ./writeup/track_1_flipped.png "Flipped Image Track 1"
[image15]: ./writeup/track_1_flipped_inverted.png "Inverted Flipped Image Track 1"
[image16]: ./writeup/track_1_original_inverted.png "Inverted Cropped Image Track 1"

[image17]: ./writeup/track_1_left.png "Left Camera View"
[image18]: ./writeup/track_1_right.png "Right Camera View"
[image19]: ./writeup/track_1_center.png "Center Camera View"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
In order to use the GPU of my MacBook Pro for Keras, I decided to use PlaidML. For that,
the ```drive.py``` file needs to be extended with:

```
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
```
Further, the PlaidML framework needs to be installed: https://www.intel.ai/plaidml/

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The file is structured in four blocks:

* Block 1: The data augmentation methods and the generator
* Block 2: Parsing the collected data, perform train, valid and test split, generator train and valid generators
* Block 3: Keras code for either load an existing model and continue training or define the model from the scratch and perform the training.
* Block 4: Visualization of the loss on the train and valid data set against the epochs.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is highly inspired by the model developes by Nvidia`s "End to End Learning for Self-Driving Cars": 

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The model i basically the same, except the Dropout layers after the fully connected Dense layers. This was introduced to reduce the effect of overfitting on the training dataset.

The model first crops the image by 70 pixels from the top and 25 pixels from the bottom using a Keras Cropping2D (line 121). This is necessary, to exclude the sky and the lower parts of the street from the image, in order to only present information relevant to steering angles to the network. Then, the model preprocesses the data by performing standardization and normalization using a Keras Lambda layer (line 121). This is needed to make each of the color channels zero-centered and have a variance of 1.

``` python
#Preprocessing
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65,320,3)))
```

Here is an example for the cropped images:

![alt text][image12]
![alt text][image13]


The model consists of 5 convolutional layers. The first 3 layers have strides of (2, 2) and a kernel of (5, 5). The strides reduce the image size but the model gets deeper in terms of more and more feature maps per convolutional layers. This helps the model to detect patterns in the input image to base the steering angle on. These were alternating equipped with Relu and Elu layers. The reasoning behind this was to introduce non-linearity and the alternating to make the model more "asymetric".

``` python
model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2), padding="valid"))
model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2), padding="valid"))
```

The last 2 layers of the convolutional layers have a stride of (1, 1), and a kernel size if (3, 3). Nvidia argumented in their paper, that this network was based on empirical experiments. I tried to add MaxPooling Layers, experimented with He initialization, adding more Layers or more feature maps on these layers, but all without positive effect.

``` python
model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid"))
model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1), padding="valid"))
model.add(Flatten())
```

After flatting the output of the convulution layers, 5 fully connected dense layers follow. Any try to apply an activation function here brought very negative effects. Also Batch Normalization did only show negative impact. Dropout was the only regularization that helped:

```python
        # Dense layers, all with dropout
        model.add(Dense(1064, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(100, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(50, kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(Dense(1))
```

I chose the mean square error (MSE) as error measure for the difference between the predicted steering angle and the ground truth steering angle:

```python
# Use adam optimizer for adaptive learning rate
model.compile(loss="mse", optimizer="adam")
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 134, 137, 140). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 154-159). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The generator for training data additionall generates mirrored versions of the input data and color-inverted versions, as well. 

In the following example, the original image is in the upper left corner. The other three images are flipped and color inverted versions of the original:

![alt text][image13]
![alt text][image14]
![alt text][image16]
![alt text][image15]

The idea behind the color-inverted was to make the model more agnostic to the color of the road. The model could drive a full lap without errors on track 1 without collecting the recovery data from sides of the track. Further, it helps to create more data and thus reduces the risk of overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 146).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded two laps on track 1, one in original direction and one in oposite direction. Additionally, I recorded one lap on track 2 in original direction.

I did not create recovering from the left and right sides of the road, as the vehicle could also manage track 1 without this data. However, I used the left and right camera as additional input images and corrected the angle by -0.2 for the right camera and +0.2 for the left camera. Here are three examples:

![Left Camera View][image17]

![Center Camera View][image18]

![Right Camera View][image19]

The left and right camera images help to simulate the recovering behaviour.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the model published by Nvidia:

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

It was clear that the model needs convolutional layers to properly detect structures in the images like the borders of the road. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contained Dropout layers after each fully connected Dense layer. Further, I extended the generator to also generate color-inverted versions of the pictures, collected a drive on track 1 against the right direction, in order to collect more training data.

Then I experimented with using Relu and Batch Normalization in the fully connected Dense layers. This had very negative effect on the learning curve, why I discarded this idea. I also experimented with Max Pooling with a stride of (1, 1) after the Convolution Layers, also without any positive effect.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Especially, in these situations:

![alt text][image81]
![alt text][image82]
![alt text][image83]
![alt text][image84]

It could be solved by additional training of the model. Overall it was trained for 50 epochs with a batch size of 200.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image12]

The left and right recovery strategy was based on the left and right camera, as explained above.


To augment the data sat, I also flipped images and angles thinking that this would help to reduce overfitting. I also did this on the color-inverted images. Examples were shown above.

Additionally, as described above, my generator also generates color-inverted versions of the flipped and original data. If augmentation is activated, one third consists of random original data, one third of color-inverted versions of original data, on third of flipped versions of flipped or original data.

After the collection process, I had 6653 number of data points. The preprocessing is done in the Keras model, as described above.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by the use of an EarlyStopping callback in Keras. The following learning curve further proofs this:

![alt text][image11]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
