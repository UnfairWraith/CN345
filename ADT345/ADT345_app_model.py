from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report









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
                                                       batch_size = 300,
                                                       class_mode = 'categorical')
    
    val_generator = train_datagen.flow_from_directory(valid_parent_directory,
                                                          target_size = (75,75),
                                                          batch_size = 46,
                                                          class_mode = 'categorical')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size = 7, #call by batches of 37
                                                     class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):

    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output neuron. 
    x = Dense(3, activation='softmax')(x) #using softmax.
    
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
    
def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size    
    
    
    
    
    
    
    
    
    

train_dir = os.path.join('data/train') #change to ur own directory.
test_dir = os.path.join('data/test') #change to ur own directory.
valid_dir = os.path.join('data/valid') #change to ur own directory.
#using windows commans to set to the default path. --> Command prompt.

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, valid_dir)


pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output  #last layer is output.

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 #hyperparameters.
 #epoch is the number of training rounds
history_TL = model_TL.fit(
train_generator,
steps_per_epoch=12,
epochs=7,
verbose=1,
validation_data = validation_generator)


plot_hist(history_TL)


tf.keras.models.save_model(model_TL,'Auntie-DOT_E0.hdf5') #will be save as this file


#HALO LORE
#Auntie DOT is UNSC's 'dumb' AI used by NOBLE team during the fall of REACH, it is one of the AI's used by ONI before the discovery of forerunner technology which enabled humanity to create 'smart' AI.