import pydot
import tensorflow as tf
from PIL import Image
from keras import regularizers
from keras.callbacks import EarlyStopping

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import matplotlib.pyplot as plt

#############################################################
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers import Dropout, Flatten, Dense
####################################################
import os
import cv2
import numpy as np

image_directory = 'Dataset/'
SIZE = 64
dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
label = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

parasitized_images = os.listdir(image_directory + 'empty/')
for i, image_name in enumerate(parasitized_images):
    # Remember enumerate method adds a counter and returns the enumerate object
    image = cv2.imread(image_directory + 'empty/' + image_name)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((SIZE, SIZE))
    image = np.array(image)
    dataset.append(image)
    label.append(0)

# Iterate through all images in Uninfected folder, resize to 64 x 64
# Then save into the same numpy array 'dataset' but with label 1

uninfected_images = os.listdir(image_directory + 'full/')
for i, image_name in enumerate(uninfected_images):
    image = cv2.imread(image_directory + 'full/' + image_name)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((SIZE, SIZE))
    image = np.array(image)
    dataset.append(image)
    label.append(1)

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)
print("X_test: ", X_test)

# Without scaling (normalize) the training may not converge.
# Normalization is a rescaling of the data from the original range
# so that all values are within the range of 0 and 1.
from keras.utils import normalize, plot_model

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# y_train = y_train.astype(int)
# y_test = y_test.astype(int)
##############################################
INPUT_SHAPE = (SIZE, SIZE, 3)

# Model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Dropout(rate=0.25))
#
# model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Dropout(rate=0.25))
#
# model.add(Flatten())
#
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(32))
# model.add(Activation('relu'))
#
# model.add(Dense(16))
# model.add(Activation('relu'))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# Создаем последовательную модель
model = Sequential()

# Добавляем сверточный слой Conv2D с 16 фильтрами, размером ядра 3x3 и активацией ReLU
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Добавляем еще один сверточный слой Conv2D с 32 фильтрами, размером ядра 3x3 и активацией ReLU
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# Добавляем слой подвыборки MaxPooling2D с окном подвыборки размером 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Добавляем еще один сверточный слой Conv2D с 128 фильтрами, размером ядра 3x3 и активацией ReLU
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Добавляем слой подвыборки MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2)))

# Добавляем плоский слой для преобразования выхода сверточных слоев в одномерный вектор
model.add(Flatten())

# Добавляем полносвязный слой с 64 нейронами и активацией ReLU
model.add(Dense(64, activation='relu'))
# Добавляем слой Dropout с вероятностью обнуления 0.5
model.add(Dropout(0.6))
# Добавляем полносвязный слой с 32 нейронами и активацией ReLU
model.add(Dense(32, activation='relu'))
# Добавляем выходной слой с 32 нейроном
model.add(Dense(16, activation='relu'))

model.add(BatchNormalization())
# Добавляем выходной слой с 1 нейроном и активацией сигмоидой
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',  # also try adam rmsprop
              metrics=['accuracy']
              )


# Визуализация модели с помощью pydot и graphviz
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print(model.summary())
###############################################################

# Создаем объект EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train,
                    y_train,
                    batch_size=8,
                    verbose=1,
                    epochs=80,
                    validation_data=(X_test, y_test),
                    # callbacks=[early_stopping],
                    shuffle=True
                    )

model.save('New_model2_80epochs_opt_rmsprop_result_img64_batchsize8_shuffle_True.h5')

print('history: ')
history_dict = history.history
print(history_dict.keys())

# plot the training and validation accuracy and loss at each epoch
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################################################################
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#########################################################################################
n = 0  # Select the index of image to be loaded for testing
# img = X_test[n]
# plt.imshow(img)
# # input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
# print("The prediction for this image is 0: ", model.predict(img))

arr_test = model.predict(X_test)
print(arr_test)


################################################################
# We can load the trained model, so we don't have to train again for 300 epochs!

# load model
# model = load_model('malaria_model.h5')

# How do we know how it is doing for parasitized vs uninfected?
################################################################

# Confusion matrix
# We compare labels and plot them based on correct or wrong predictions.
# Since sigmoid outputs probabilities we need to apply threshold to convert to label.

"""
mythreshold = 0.908
from sklearn.metrics import confusion_matrix

y_pred = (model.predict(X_test) >= mythreshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""

# Check the confusion matrix for various thresholds. Which one is good?
# Need to balance positive, negative, false positive and false negative.
# ROC can help identify the right threshold.
##################################################################
"""
Receiver Operating Characteristic (ROC) Curve is a plot that helps us 
visualize the performance of a binary classifier when the threshold is varied. 
"""
# ROC
"""
from sklearn.metrics import roc_curve

y_preds = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
"""

"""
#One way to find the best threshold once we calculate the true positive 
and false positive rates is ...
The optimal cut off point would be where “true positive rate” is high 
and the “false positive rate” is low. 
Based on this logic let us find the threshold where tpr-(1-fpr) is zero (or close to 0)
"""

# i = np.arange(len(tpr))
# roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
# ideal_roc_thresh = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]  # Locate the point where the value is close to 0
# print("Ideal threshold is: ", ideal_roc_thresh['thresholds'])

# Now use this threshold value in the confusion matrix to visualize the balance
# between tp, fp, fp, and fn


# AUC
# Area under the curve (AUC) for ROC plot can be used to understand hpw well a classifier
# is performing.
# % chance that the model can distinguish between positive and negative classes.
"""
from sklearn.metrics import auc

auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)
"""

#########################################
