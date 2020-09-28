"""
Convolutional Neural Network Models
"""

from keras import layers, models

def default(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (1, 1), activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((1, 1)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model
