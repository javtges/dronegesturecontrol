# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


## James things
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class LeapGesture(Dataset):
        def __init__(self, data, target, transform=None):
            self.data = data # This will be the 1050*5 x 3 x 16 x 112 x 112 data?
            self.target = target#torch.FloatTensor(target).long #torch.from_numpy(target).long
            # print(self.data)
            # print(self.target)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self,index):
            x = self.data[index]
            y = self.target[index]

            return x, y


def parse_data():

    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('../leapgestrecog/leapGestRecog/00/'):
        if not j.startswith('.'): # If running this code locally, this is to 
                                # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    lookup


    print(lookup)

    key_list = list(lookup.keys())
    val_list = list(lookup.values())

    x_data = []
    y_data = []
    IMG_SIZE = 150
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(0, 10): # Loop over the ten top-level folders
        for j in os.listdir('../leapgestrecog/leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('../leapgestrecog/leapGestRecog/0' + 
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    path = '../leapgestrecog/leapGestRecog/0' + str(i) + '/' + j + '/' + k
                    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                    arr = np.array(img)
                    x_data.append(arr) 
                    count = count + 1
                y_values = np.full((count, 1), lookup[j]) 
                y_data.append(y_values)
                datacount = datacount + count
    x_data = np.array(x_data, dtype = 'float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

    # y_data=to_categorical(y_data)
    x_data = x_data.reshape((datacount, 1, IMG_SIZE, IMG_SIZE))
    x_data = x_data/255

    print(x_data.shape)
    print(y_data.shape)
    print(y_data)

    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,random_state=42)

    train_loader = torch.utils.data.DataLoader(dataset=LeapGesture(x_train, y_train), batch_size=128, shuffle=True, pin_memory=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=LeapGesture(x_test, y_test), batch_size=128, shuffle=True, pin_memory=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
# check some image
    parse_data()

    # fig,ax=plt.subplots(5,2)
    # fig.set_size_inches(15,15)
    # for i in range(5):
    #     for j in range (2):
    #         l=rn.randint(0,len(y_data))
    #         ax[i,j].imshow(x_data[l])
    #         ax[i,j].set_title(reverselookup[y_data[l,0]])
            
    # plt.tight_layout()
    # plt.show()




