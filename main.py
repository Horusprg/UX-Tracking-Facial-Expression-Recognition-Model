
'''
All the code was developed by Flávio Moura, a scientific research from the Federal University of Pará and member of the 
Operational Research Laboratory. This repo aims to train multiple models for the task of facial expression recognition.
The main objetive of the development of that models is to be deployed in monitoring tools from the UX-Tracking Framework,
also developed by Flávio Moura. For that reason, the models developed was evaluated focusing on Precision metric and low
computational cost architectures of AI models was selected.

For more informations about:
    - Development of the models
    - Architecture of the UX-Tracking framework
    - Scientif Research behind this code and the fully framework
    - Colaboration in a scientific research

Please, contact Flávio Moura in the email: flavio.moura@itec.ufpa.br
'''

import tensorflow as tf
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from utils import load_and_tune, model_select

# Data settings
data_dir = 'AffectNet'
img_height, img_width = 96, 96
batch_size = 256

train_ds, val_ds = load_and_tune(data_dir=data_dir, 
                                 img_height=img_height,
                                 img_width=img_width,
                                 batch_size=batch_size)

# Model selection (EfficientNet, ...)

model = model_select(model='EfficientNet1',
                     img_height=img_height,
                     img_width=img_width)

# Train settings
epochs = 100

# Exponential learning rate
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-1/epochs)

# Callbacks
early_stopping = EarlyStopping(monitor='val_mse', patience=20, verbose=0, mode='min')
model_checkpoint = ModelCheckpoint('best_lstm.h5', monitor='val_mse', mode='min', save_best_only=True, verbose=0)
reduce_lr = LearningRateScheduler(schedule=scheduler, verbose=1)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=epochs, validation_data=val_ds)
