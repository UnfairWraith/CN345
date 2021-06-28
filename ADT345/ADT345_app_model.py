from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools








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
    
    
    
    
    
    
    
    
    
    

train_dir = os.path.join('../dataset_ADT/train') #change to ur own directory.
test_dir = os.path.join('../dataset_ADT/test') #change to ur own directory.
valid_dir = os.path.join('../dataset_ADT/valid') #change to ur own directory.
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

Y_pred = model_TL.predict(validation_generator, 9)
y_pred = np.argmax(Y_pred, axis=1)

plot_confusion_matrix(confusion_matrix(validation_generator.classes, y_pred),['3','4','5'], normalize=True)

print('')
print('')
print('')
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))


print('Classification Report')
print(metrics.classification_report(validation_generator.classes, y_pred))

plot_hist(history_TL)



tf.keras.models.save_model(model_TL,'ADT345.hdf5') #will be save as this file
