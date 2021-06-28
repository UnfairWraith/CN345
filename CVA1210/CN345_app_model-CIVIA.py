from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from sklearn import metrics










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
    
    
    
    

train_dir = os.path.join('assets/train') #change to ur own directory.
test_dir = os.path.join('assets/test') #change to ur own directory.
valid_dir = os.path.join('assets/valid') #change to ur own directory.
#using windows commans to set to the default path. --> Command prompt.

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, test_dir, valid_dir)


pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed10')
last_output = last_layer.output  #last layer is output.

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 #hyperparameters.
 #epoch is the number of training rounds
history_TL = model_TL.fit(
train_generator,
steps_per_epoch=10,
epochs=18,
verbose=1,
validation_data = validation_generator)

Y_pred = model_TL.predict(validation_generator, 800)
y_pred = np.argmax(Y_pred, axis=1)

print('')
print('')
print('')
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('')
print('')
print('Classification Report')
print(metrics.classification_report(validation_generator.classes, y_pred))

plot_hist(history_TL)


tf.keras.models.save_model(model_TL,'CN345_CIVIA-TI.hdf5') #will be save as this file

