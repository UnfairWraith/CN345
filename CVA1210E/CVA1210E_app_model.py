from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import time
import datetime
from tensorflow import keras
from sklearn import metrics
import itertools

from tensorflow.keras.preprocessing import image



NUM_CLASSES = 10





def create_model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    efficientNet = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    
    #inputs = tf.keras.layers.Input(shape=(380, 380, 3))
    #efficientNet = tf.keras.applications.EfficientNetB4(weights="imagenet", include_top=False, input_tensor=inputs)
    
    #inputs = tf.keras.layers.Input(shape=(600, 600, 3))
    #efficientNet = tf.keras.applications.EfficientNetB7(weights="imagenet", include_top=False, input_tensor=inputs)
    
    #print("Number of layers in the base model: ", len(efficientNet.layers))
    #print(efficientNet.summary())

    #x = efficientNet.layers[-2].output

    # Freeze the pretrained weights
    efficientNet.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(efficientNet.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)

    output = Dense(units=NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=efficientNet.input, outputs=output)

    # Freeze all the layers before the target layer
    '''
    for layer in model.layers[:200]:
      layer.trainable =  False
    '''
    
    for layer in model.layers:
      layer.trainable =  False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print("Number of layers in modified model: ", len(model.layers))
    #print(len(model.trainable_variables))
    #print(model.summary())
    return model



def image_gen_w_aug(train_parent_directory, test_parent_directory, valid_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255, #scaling
                                      rotation_range = 30,   #rotation 
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.15) #split to 15% of validation set.
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (75,75),
                                                       batch_size = 337,
                                                       class_mode = 'categorical')
    
    val_generator = train_datagen.flow_from_directory(valid_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 80,
                                                          class_mode = 'categorical')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size = 20, #call by batches of 37
                                                     class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(10, activation='softmax')(x) #using softmax.
    
    model = Model(pre_trained_model.input, x)
    
    return model

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
 
#Add Normalization Option
   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print("Normalized confusion matrix")
   else:
     print('Confusion matrix, without normalization')
 
# print(cm)
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else 'black')
 
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label') 
   plt.show()
     
    
    
def finetune_model(model):
    # Unfreeze the top 20 layers, and leave BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

   
    
    
    
    
    
    

train_dir = os.path.join('../dataset_CVA1210/train') #change to ur own directory.
test_dir = os.path.join('../dataset_CVA1210/test') #change to ur own directory.
valid_dir = os.path.join('../dataset_CVA1210/valid') #change to ur own directory.
#using windows commans to set to the default path. --> Command prompt.

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, valid_dir)


model = create_model()
start_time = time.time()

hist1 = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 10)

end_time = time.time()

print("Total training time: {} seconds".format(end_time - start_time))
#print(model.summary())
plot_hist(hist1)



def finetune_model(model):
    # Unfreeze the top 20 layers, and leave BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model= finetune_model(model)

hist2 = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 10)


plot_hist(hist2)






tf.keras.models.save_model(model,'CVA1210E.hdf5') #will be save as this file




#failed Attempts


# EFFICIENT NET BEE ZERO
#import efficientnet.keras as efn
#model = efn.EfficientNetB0(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
#for layer in model.layers:
#    layer.trainable = False
#x = model.output
#x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
#predictions = Dense(10, activation="softmax")(x)
#model_final = Model(input = model.input, output = predictions)  
#model_final.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])   
#eff_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 12, epochs = 10)
#plot_hist(eff_history) # PLOT Histogram
#tf.keras.models.save_model(model,'CN345_CIVIA.hdf5') #will be save as this file



#VGG-16

#from tensorflow.keras.applications.vgg16 import VGG16
#base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
#include_top = False, # Leave out the last fully connected layer
#weights = 'imagenet')
#for layer in base_model.layers:
#    layer.trainable = False
#x = Flatten()(base_model.output)
# Add a fully connected layer with 512 hidden units and ReLU activation
#x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
#x = Dropout(0.2)(x)
# Add a final sigmoid layer for classification
#x = Dense(10, activation='softmax')(x)
#model = tf.keras.models.Model(base_model.input, x)  
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['acc'])
#vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)
#plot_hist(vgghist) # PLOT Histogram
#tf.keras.models.save_model(model,'CN345_CIVIA-VGG16.hdf5') #will be save as this file



#RESNET50

#from tensorflow.keras.applications import ResNet50
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
#base_model = Sequential()
#base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
#base_model.add(Dense(10, activation='softmax'))
#base_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
#resnet_history = base_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 12, epochs = 10)
#plot_hist(resnet_history)
#tf.keras.models.save_model(base_model,'CN345_CIVIA-VGG16.hdf5')


