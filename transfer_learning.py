import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Directory with training horse pictures
train_horse_dir = os.path.join('./horse-or-human/horses')
validation_horse_dir = os.path.join('./validation-horse-or-human/horses')

# Directory with training human pictures
train_human_dir = os.path.join('./horse-or-human/humans')
validation_human_dir = os.path.join('./validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])
validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])
validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])
print('total training horse images:{}'.format(len(os.listdir(train_horse_dir))))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
BATCH_SIZE = 64

# # Flow training images in batches of 64 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './horse-or-human/',  # This is the source directory for training images
        target_size=(160, 160),  # All images will be resized to 300x300
        batch_size=64,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    './validation-horse-or-human/', # This is the source directory for validation images
    target_size=(160,160),
    batch_size=64,
    class_mode='binary'
)

image_batch = next(train_generator)
validation_batche = next(validation_generator)
print(image_batch[0].shape)

IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
# base_model.summary()

# To generate predictions from the block of features, average over the spatial 5x5 spatial locations,
# using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# add a dense layer for binary classification
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# pack all layer together
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
model.summary()

base_learning_rate = 0.0001
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
len(model.trainable_variables)

initial_epochs = 10

num_train = 1027
num_validation = 256

steps_per_epoch = num_train // BATCH_SIZE
validation_steps= num_validation // BATCH_SIZE
print(validation_batche[1].shape)
loss0,accuracy0 = model.evaluate_generator(validation_generator, steps=validation_steps)
print(10*"=", "initial loss: {0}, initial accuracy: {1}".format(loss0, accuracy0))
model.save('transfer_model.h5')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=8,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=validation_steps
)

# plot the learning curve
acc =[]
acc += history.history['accuracy']
val_acc=[]
val_acc += history.history['val_accuracy']
loss=[]
loss += history.history['loss']
val_loss=[]
val_loss += history.history['val_loss']

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

plt.savefig('transfer_learn.png')

# Fine tuning

# Un-freeze the top layers of the model
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 100
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_tuning =  model.fit_generator(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=total_epochs,
      initial_epoch = history.epoch[-1],
      verbose=1,
      validation_data=validation_generator,
      validation_steps=validation_steps
)

acc += history_tuning.history['accuracy']
val_acc += history_tuning.history['val_accuracy']

loss += history_tuning.history['loss']
val_loss += history_tuning.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('fine_tune.png')