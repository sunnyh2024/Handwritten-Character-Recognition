import matplotlib.pyplot as plt
import cv2
import numpy as np
#from sgd import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from SGDv2 import SGDv2

# Used these resource for help:
# https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
# https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# https://www.analyticsvidhya.com/blog/2021/06/building-a-convolutional-neural-network-using-tensorflow-keras/

# read the character training data
char_csv = pd.read_csv('./data/A_Z_Handwritten_Data2.csv').astype('float32')

# split the data into char_images and char_labels
char_images = char_csv.drop('0', axis=1)
char_labels = char_csv['0']
print(char_labels)

# split the data again into train and test
train_val_images, test_images, train_val_labels, test_labels = train_test_split(char_images, char_labels, test_size=0.2)
train_images, val_images, train_labels, val_labels = train_test_split(train_val_images, train_val_labels, train_size=0.75, test_size = 0.25)
train_images = np.reshape(train_images.values, (train_images.shape[0], 28, 28))
val_images = np.reshape(val_images.values, (val_images.shape[0], 28, 28))
test_images = np.reshape(test_images.values, (test_images.shape[0], 28, 28))

# all valid characters
ALPHA_DIC = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7', 34:'8',35:'9'}

# graph the frequencies of each label and displays it
labels_int = np.int0(char_labels)
count = np.zeros(36, dtype='int')
for i in labels_int:
    count[i] += 1

nums = []
for i in ALPHA_DIC.values():
    nums.append(i)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.barh(nums, count)

plt.xlabel('Number of elements ')
plt.ylabel('Numbers')
plt.grid()
plt.show()

# example character_images
fig, ax = plt.subplots(3, 3, figsize=(10,10))
axes = ax.flatten()

for i in range(9):
    _, shu = cv2.threshold(train_images[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(train_images[i], (28, 28)), cmap='Greys')
plt.show()

# reshaping the data for the Sequential model
train_Images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
val_Images = val_images.reshape(val_images.shape[0], val_images.shape[1], val_images.shape[2], 1)
test_Images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

train_labelOHE = to_categorical(train_labels, num_classes=36, dtype='int')
val_labelOHE = to_categorical(val_labels, num_classes=36, dtype='int')
test_labelOHE = to_categorical(test_labels, num_classes=36, dtype='int')
print('New shape of train dig_labels: ', train_labelOHE.shape)
print('New shape of validation dig_labels: ', val_labelOHE.shape)
print('New shape of test dig_labels: ', test_labelOHE.shape)

#creating the model
#A Sequential model is appropriate for a plain stack of layers where each layer 
#has exactly one input tensor and one output tensor.
model = Sequential()

# Runs through a series of Conv2D and MaxPool2D layers to learn about features from most specific
# to least specific while making the size of the data smaller so it's easier to deal with
# This order of layers has been decided through online research of what works best as well as some trial
# and error to see how the model improves with changes.
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
#Uses the output of the flattening operation to then create a fully connected neural network
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(36,activation ="softmax"))

# training the model
print('started training')
#After significant testing, these are the hyperparameters we decided to use.
model.compile(optimizer = SGD(learning_rate=0.001, momentum=0.66), loss='categorical_crossentropy', metrics=['accuracy'])
# Reduces the learning rate as the model fits the data
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
#Strops training when the metric stops improving
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
history = model.fit(train_Images, train_labelOHE, epochs=5, callbacks = [reduce_lr, early_stop], validation_data = (val_images, val_labelOHE))
test_results = model.evaluate(test_Images, test_labelOHE, 128)
print('test results:', test_results)

# saving and getting some info from the model (specifically accuracy and loss)
# The saved file can be used for the gui
model.summary()
model.save(r'./present.h5')

print("The validation accuracy is :", history.history['val_accuracy'][-1])
print("The training accuracy is :", history.history['accuracy'][-1])
print("The validation loss is :", history.history['val_loss'][-1])
print("The training loss is :", history.history['loss'][-1])

