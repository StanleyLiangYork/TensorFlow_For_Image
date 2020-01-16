import tensorflow as tf
import numpy as np
import os

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

CLASS_NAME = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
model = tf.keras.models.load_model('./mnist.h5')
print(test_images.shape)
predicts = model.predict(test_images)

for i in range(0,10):
    predict_idx = np.argmax(predicts[i])
    true_idx = test_labels[i]
    print("The predicted label is: {0}, the ground true is: {1}".format(CLASS_NAME[predict_idx], CLASS_NAME[true_idx]))


# predict label of raw image

# use this line if the saved model is in HDF5 format
#model_cnn = tf.keras.models.load_model('my_model.h5')

# use this line if the saved model is in TensorFlow format
model_cnn = tf.keras.models.load_model('horse_human_TF_model')
model_cnn.summary()

# # load images files
predict_dir = os.path.join('./test_data')
predict_files = os.listdir(predict_dir)
print(predict_files)
#
for fn in predict_files:
    path = './test_data/' + fn
    print(path)
    img = tf.keras.preprocessing.image.load_img(path, target_size=(300, 300))
    x = tf.keras.preprocessing.image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    # print(x.shape)
    images = np.vstack([x])
    classes = model_cnn.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")