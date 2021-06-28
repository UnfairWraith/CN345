import time
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


NUM_CLASSES = 11
input_size = (600, 600)

TRAIN_PATH = os.path.join('assets/train')
VALID_PATH = os.path.join('assets/test')
TEST_PATH = os.path.join('assets/valid')

MODEL_NAME = "CN345_CIVIA-EfficientNetB4"

def load_image(file):
    img = image.load_img(file, target_size=input_size)
    img_array = image.img_to_array(img)
    #import pdb; pdb.set_trace()
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

print(tf.version.VERSION)



def create_model():
    #inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    #efficientNet = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    
    inputs = tf.keras.layers.Input(shape=(380, 380, 3))
    efficientNet = tf.keras.applications.EfficientNetB4(weights="imagenet", include_top=False, input_tensor=inputs)
    
    #inputs = tf.keras.layers.Input(shape=(600, 600, 3))
    #efficientNet = tf.keras.applications.EfficientNetB7(weights="imagenet", include_top=False, input_tensor=inputs)
    
    #print("Number of layers in the base model: ", len(efficientNet.layers))
    #print(efficientNet.summary())

    #x = efficientNet.layers[-2].output

    # Freeze the pretrained weights
    efficientNet.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(efficientNet.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    output = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=efficientNet.input, outputs=output)

    # Freeze all the layers before the target layer
    '''
    for layer in model.layers[:200]:
      layer.trainable =  False
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Number of layers in modified model: ", len(model.layers))
    print(len(model.trainable_variables))
    #print(model.summary())
    return model


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=TRAIN_PATH, target_size=input_size, batch_size=337)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=VALID_PATH, target_size=input_size, batch_size=80)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=TEST_PATH, target_size=input_size, batch_size=20, shuffle=False)
print(MODEL_NAME)

model = create_model()

start_time = time.time()

hist1 = model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            #callbacks=[tensorboard_callback],
            epochs=6,
            verbose=1
)

end_time = time.time()

#model.save("/content/drive/My Drive/CS_231A/saved_models/" + MODEL_NAME)

print("Total training time: {} seconds".format(end_time - start_time))
print(model.summary())
plot_hist(hist1)

#model = tf.keras.models.load_model("/content/drive/My Drive/CS_231A/saved_models/train2_efficient_depth_data")
model.save("/content/drive/My Drive/CS_231A/saved_models/EfficientNetB7_base_stage1")


print(model.summary())

def finetune_model(model):
    # Unfreeze the top 20 layers, and leave BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model= finetune_model(model)

hist2 = model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=50,
            verbose=2
)

plot_hist(hist2)
tf.keras.models.save_model(model,'CN345_CIVIA_E4.hdf5') #will be save as this file


test_labels = test_batches.classes
results = model.evaluate(test_batches, batch_size=128)
print("test loss, test acc:", results)




print(model.summary())