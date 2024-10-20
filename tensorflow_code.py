import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Flatten, AveragePooling2D
from keras.models import Model

classes = []
class_counter = 0

for dirname, _, filenames in os.walk(r'C:\Users\Ecoloop\PycharmProjects\pythonProject\Fresh_fruit_nonfruit\dataset\train'):
    if dirname.endswith('/'):
        continue
    else:
        classes.append({dirname.split('/')[-1]: 0})
    file_count = 0
    for filename in filenames:
        file_count += 1
    classes[class_counter][dirname.split('/')[-1]] = file_count
    class_counter += 1

print('{:<15} {:<15}'.format('Class', 'Number of instances'))
print()
for d in classes:
    [(k, v)] = d.items()
    print('{:<15} {:<15}'.format(k, v))

counts = []
labels = []
for d in classes:
    [(k, v)] = d.items()
    labels.append(k)
    counts.append(v)

plt.figure()
plt.bar(range(len(counts)), counts, color = ['green', 'orange', 'grey', 'yellow', 'brown', 'red'], alpha = .7)
plt.xticks(range(len(counts)), labels, rotation = 30)
plt.title('Count of each label in our training data')
plt.show()
TRAIN_PATH = r'C:\Users\Ecoloop\PycharmProjects\pythonProject\Fresh_fruit_nonfruit\dataset\train'
TEST_PATH = r'C:\Users\Ecoloop\PycharmProjects\pythonProject\Fresh_fruit_nonfruit\dataset\test'
datagen = ImageDataGenerator(
    rotation_range = 30,
    zoom_range = .3,
    horizontal_flip = True,
    vertical_flip = True,
    validation_split = .3
)

train_ds = datagen.flow_from_directory(
    directory = TRAIN_PATH,
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    subset = 'training'
)

validation_ds = datagen.flow_from_directory(
    directory = TRAIN_PATH,
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    subset = 'validation'
)

vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
vgg16.trainable = False
X_input = Input(shape = (224, 224, 3))
X = vgg16(X_input)
X = AveragePooling2D(pool_size = (3, 3), strides = 2, padding = 'valid',name = 'AvgPool2D')(X)
X = Flatten(name = 'Flatten')(X)
X = Dense(200, activation = 'relu', name = 'Dense1')(X)
X = Dropout(.1)(X)
X = Dense(100, activation = 'relu', name = 'Dense2')(X)
X = Dropout(.1)(X)
X = Dense(6, activation = 'softmax', name = 'Dense3')(X)

model = Model(inputs = X_input, outputs = X, name = 'Fruit_Classifer')

print(model.summary())
optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

hist = model.fit(train_ds, validation_data = validation_ds, epochs = 8, batch_size = 32)
model.save("model_vvg.h5")
model.summary()
