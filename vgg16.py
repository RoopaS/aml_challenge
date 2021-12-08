import datetime
import tensorflow as tf
import keras
import pandas as pd
from pathlib import Path

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout

# TODO Find a way to turn off that red debugging spam from tensorflow, this does not work
tf.get_logger().setLevel('WARN')

print(f'Using GPU {tf.test.gpu_device_name()}')

'''

Added parameters to match with VGG16 and for augmenting data

'''

# Global params and constants
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 50
EPOCHS = 3
TRAIN_IMAGES_PATH = r'./dataset/train_set_labelled'
TEST_IMAGES_PATH = r'./dataset/test_set'
TRAIN_LABELS_PATH = r'./dataset/train_labels.csv'
PREDICTIONS_PATH = r'predictions.csv'
NUM_EXAMPLES = len(list(Path(TRAIN_IMAGES_PATH).rglob('*.jpg')))
NUM_CLASSES = len(list(Path(TRAIN_IMAGES_PATH).iterdir()))
print(f'Num classes: {NUM_CLASSES}  num samples: {NUM_EXAMPLES}')

# Generators allow to get the data in batches without having to worry about the memory
train_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode="nearest"
#     featurewise_center=True,
#     featurewise_std_normalization=True
)
val_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale = 1./255,
)

train_gen = train_generator.flow_from_directory(
    directory=TRAIN_IMAGES_PATH,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=True,
    subset='training'
)
validation_gen = val_generator.flow_from_directory(
    directory=TRAIN_IMAGES_PATH,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=True,
    subset='validation'
)
test_gen = val_generator.flow_from_directory(
    directory=TEST_IMAGES_PATH,
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=(WIDTH, HEIGHT),
    shuffle=False
)


def get_model() -> keras.Model:
    """
    Build, compile and return the model
    """
    model = Sequential()
    model.add(Input(shape=(WIDTH, HEIGHT, 3)))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
#     model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
#     model.add(Dropout(0.2))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics='accuracy')
    return model


# def make_predictions(model: keras.Model, test_gen: ImageDataGenerator):
#     """
#     Output a CSV with model's predictions on test set that is ready to be submitted to Kaggle.
#     The file will be created in the main directory of the project, named 'predictions <current_time>'
#     """
#     predictions = model.predict(test_gen, verbose=True, batch_size=BATCH_SIZE)
#     # Get names of test files in the same order they were used for predictions
#     file_names = list(map(lambda x: x.split('\\')[1], test_gen.filenames))
#     # Obtain final labels for predictions, add one since classes start from one
#     predictions = predictions.argmax(axis=1) + 1
#     result = pd.DataFrame({'img_name': file_names, 'label': predictions})
#     result = result.set_index('img_name')
#     # Save the CSV file to main project directory
#     result.to_csv(f'predictions {datetime.datetime.now().strftime("%d-%m-%Y %Hh %Mm %Ss")}.csv')


model = get_model()
model.summary()
model.fit(
    train_gen,
    validation_data=validation_gen,
#     steps_per_epoch=10,
#     validation_steps=1,
    epochs=EPOCHS,
    shuffle=True,
    verbose=True,
)

# score = model.evaluate(test_gen, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# make_predictions(model=model, test_gen=test_gen)

# '''
# Import the VGG16 library and add preprocessing layer to the front of VGG
# ''' 
# from tensorflow.keras.applications.vgg16 import VGG16
# from keras.models import Model

# vgg16 = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)

# # to not train existing weights
# for layer in vgg16.layers:
#     layer.trainable = False
    
# model = Model(inputs=vgg16.input)
# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(
#     train_gen,
#     validation_data=validation_gen,
# #     steps_per_epoch=10,
# #     validation_steps=1,
#     epochs=EPOCHS,
#     verbose=True,
# )