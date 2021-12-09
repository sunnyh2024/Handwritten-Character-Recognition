import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Used these resource for help:
# https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
# https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# https://www.analyticsvidhya.com/blog/2021/06/building-a-convolutional-neural-network-using-tensorflow-keras/

# read the digit training data
digits_csv = pd.read_csv('./data/mnist_train.csv').astype('float32')
digits_test_csv = pd.read_csv('./data/mnist_test.csv').astype('float32')


print(digits_csv.head(10))

# split the data into labels and images
images = digits_csv.drop('label', axis=1)
labels = digits_csv['label']
test_images = digits_test_csv.drop('label', axis=1)
test_labels = digits_test_csv['label']
print(images)

# split the data again into train and test
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)
train_images = np.reshape(train_images.values, (train_images.shape[0], 28, 28))
val_images = np.reshape(val_images.values, (val_images.shape[0], 28, 28))
test_images = np.reshape(test_images.values, (test_images.shape[0], 28, 28))

print(train_images.shape)
print(val_images.shape)
print(test_images.shape)

# all valid digits
DIG_DIC = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# graph the frequencies of each label
labels_int = np.int0(labels)
count = np.zeros(10, dtype='int')
for i in labels_int:
    count[i] += 1

nums = []
for i in DIG_DIC.values():
    nums.append(i)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.barh(nums, count)

plt.xlabel('Number of elements ')
plt.ylabel('Numbers')
plt.grid()
plt.show()

# example images
fig, ax = plt.subplots(3, 3, figsize=(10,10))
axes = ax.flatten()

for i in range(9):
    _, shu = cv2.threshold(train_images[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(train_images[i], (28, 28)), cmap='Greys')
plt.show()

# reshaping the data for the Sequential model
train_Images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
val_images = val_images.reshape(val_images.shape[0], val_images.shape[1], val_images.shape[2], 1)
test_Images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

train_labelOHE = to_categorical(train_labels, num_classes=10, dtype='int')
val_labelOHE = to_categorical(val_labels, num_classes=10, dtype='int')
test_labelOHE = to_categorical(test_labels, num_classes=10, dtype='int')
print('New shape of train labels: ', train_labelOHE.shape)
print('New shape of validation labels: ', val_labelOHE.shape)
print('New shape of test labels: ', test_labelOHE.shape)

#creating the model
#A Sequential model is appropriate for a plain stack of layers where each layer 
#has exactly one input tensor and one output tensor.
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(10,activation ="softmax"))

# training the model
print('started training')
model.compile(optimizer = SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_Images, train_labelOHE, epochs=5,  validation_data = (val_images, val_labelOHE))
test_results = model.evaluate(test_Images, test_labelOHE, 128)
print('test results:', test_results)

# saving and getting some info from the model (specifically accuracy and loss)
model.summary()
model.save(r'./digit_recognize.h5')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

