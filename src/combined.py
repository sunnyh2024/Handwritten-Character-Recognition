import matplotlib.pyplot as plt
import cv2
import numpy as np
from sgd import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# read the digit training data
char_csv = pd.read_csv('./data/A_Z_Handwritten_Data.csv').astype('float32')

# split the data into dig_labels and dig_images
char_images = char_csv.drop('0', axis=1)
char_labels = char_csv['0']
print(char_labels)

# split the data again into train and test
train_val_images, test_images, train_val_labels, test_labels = train_test_split(char_images, char_labels, test_size=0.2)
train_images, val_images = train_test_split(train_val_images, train_size=0.75)
train_labels, val_labels = train_test_split(train_val_labels, train_size=0.75)
train_images = np.reshape(train_images.values, (train_images.shape[0], 28, 28))
val_images = np.reshape(val_images.values, (val_images.shape[0], 28, 28))
test_images = np.reshape(test_images.values, (test_images.shape[0], 28, 28))

# all valid digits
ALPHA_DIC = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7', 34:'8',35:'9'}

# graph the frequencies of each label
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

# example dig_images
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

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(36,activation ="softmax"))

# training the model
print('started training')
model.compile(optimizer = SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_Images, train_labelOHE, epochs=10,  validation_data = (val_Images, val_labelOHE))
test_results = model.evaluate(test_Images, test_labelOHE, 128)
print('test results:', test_results)

# saving and getting some info from the model (specifically accuracy and loss)
model.summary()
model.save(r'./combined.h5')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

