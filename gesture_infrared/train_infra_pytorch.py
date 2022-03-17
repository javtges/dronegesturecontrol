from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import datetime


import infrared_keras


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 10

# model = Sequential()
# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,1)))
# model.add(MaxPooling2D(pool_size=(2,2)))


# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

# model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(10, activation = "softmax"))



class infra_classifier(nn.Module):

    def __init__(self):
        super(infra_classifier,self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,5), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(64, 96, kernel_size=(3,3), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv4 = nn.Conv2d(96, 96, kernel_size=(3,3), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.dense = nn.Linear(7776, 512)
        self.dense2 = nn.Linear(512, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.dense(x))

        x = self.relu(self.dense2(x))

        probs = self.softmax(x)

        return probs


def saveModel(acc): 
    ct = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")
    path = "./network_infra_pytorch_" + ct + str(acc) + "_.pth"
    torch.save(model.state_dict(), path) 

if __name__ == "__main__":

    train_loader, test_loader = infrared_keras.parse_data()

    model = infra_classifier()

    model = nn.DataParallel(model)
    model.to(device)

    if(not torch.cuda.is_available()):
        print("???")
        from torchsummary import summary
        summary(model, (1, 150, 150), batch_size=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    
    print("did it make it here?")

    epochs = 20

    hasprint = True

    writer = SummaryWriter()
    best_accuracy = 0

    for epoch in range(1, epochs+1):
        print("\nEpoch : %d"%epoch)
        running_train_loss = 0.0
        running_val_loss = 0.0
        running_accuracy = 0.0
        total = 0
        # Loop through all the data
        for i, data in enumerate(train_loader,0):
            # print("batch?")
            # print(i)
            # print(data[0].shape)
            inputs, labels = data
            if hasprint:
                print("inputs and labels shapes")
                print(inputs.shape)
                print(labels.shape)
                hasprint = False
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # print(outputs.shape)
            # print(labels.shape)
            labels = torch.squeeze(labels)
            # print(labels.shape)


            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss/len(train_loader)

        with torch.no_grad(): 
            model.eval() 
            for data in test_loader:
                inputs, outputs = data 
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    outputs = outputs.cuda()
                predicted_outputs = model(inputs)
                # print(predicted_outputs)
                # print(outputs)

                # perhaps get rid of this
                outputs = torch.squeeze(outputs)
                val_loss = criterion(predicted_outputs, outputs)
                
                # The label with the highest value will be our prediction 
                _, predicted = torch.max(predicted_outputs, 1) 
                # print(predicted)
                running_val_loss += val_loss.item()
                total += outputs.size(0) 
                running_accuracy += (predicted == outputs).sum().item() 

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(test_loader)
                    
            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
            accuracy = (100 * running_accuracy / total)     
    
            # Save the model if the accuracy is the best 
            if accuracy > best_accuracy: 
                accuracy = round(accuracy, 3)
                saveModel(accuracy) 
                best_accuracy = accuracy 
            
            # Print the statistics of the epoch 
            writer.add_scalar("Train Loss", train_loss_value, epoch)
            writer.add_scalar("Validation Loss", val_loss_value, epoch)
            writer.add_scalar("Test Accuracy", accuracy, epoch)
            print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %.4f %%' % (accuracy))

        # # https://androidkt.com/calculate-total-loss-and-accuracy-at-every-epoch-and-plot-using-matplotlib-in-pytorch/
        # # https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model
    writer.flush()
    writer.close()




