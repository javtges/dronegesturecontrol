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
import model_C3D

sys.path.append('../nvGesture')
import readdata


# Importing pretrained C3D Model from pickle
# Setting up the network to work with my video data
# Currently, (240, 320, 3, 80) x 1050 training
# This network takes an input of 3x16x112x112
# Todo: put this in multiple files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 25
batch_size = 10

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        # add more layers in the future to adapt to the right dimensions
        # conv_input = tf.reshape(data,[-1,8,112,112,3]);

        # self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # same
        self.conv1 = nn.Conv3d(16, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # same
        self.norm1a = nn.LayerNorm([64, 112, 112, 3])
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # same

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm2a = nn.LayerNorm([128, 112, 56, 1])
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)) # same

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm3a = nn.LayerNorm([256, 56, 28, 1])
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm3b = nn.LayerNorm([256, 56, 28, 1])
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)) # same (2,2,2)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm4a = nn.LayerNorm([512, 28, 14, 1])
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm4b = nn.LayerNorm([512, 28, 14, 1])
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)) # same

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm5a = nn.LayerNorm([512, 14, 7, 1])
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)) # same
        self.norm5b = nn.LayerNorm([512, 14, 7, 1])
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=(0, 1, 1)) # This was removed
        self.fc6_size = 512 * 14 * 7 * 1
        self.fc6 = nn.Linear(self.fc6_size, 512)  # 50176
        self.fc7 = nn.Linear(512, 512)
        # self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        # print("initial shape!")
        # print(x.shape)
        # test = x[:,1,:,:,:]
        # print(test.shape)
        h = self.relu(self.conv1(x))
        # print("AFTER CONV1")
        # print(h.shape)
        h = self.relu(self.norm1a(h))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        # print("AFTER CONV2")
        # print(h.shape)
        h = self.relu(self.norm2a(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.norm3a(h))
        h = self.relu(self.conv3b(h))
        h = self.relu(self.norm3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.norm4a(h))
        h = self.relu(self.conv4b(h))
        h = self.relu(self.norm4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        # print("conv5a output shape")
        # print(h.shape)
        # h = h.view(-1, 512, 7, 7, 1)
        h = self.relu(self.norm5a(h))
        h = self.relu(self.conv5b(h))
        # print("conv5b output shape")
        # print(h.shape)
        h = self.relu(self.norm5b(h))
        # h = self.pool5(h) # This was removed
        # print("norm5b output shape")
        # print(h.shape)
        # print("first forward")

        h = h.view(-1, self.fc6_size) # 50176
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(h)
        probs = h

        return probs

    # Adapting C3D to my data

class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        
        # self.ltsm = nn.LSTMCell(487, 487)

        self.fc1 = nn.Linear(487,25) # should be 512*2?
        # self.lin1 = nn.Linear(487, 256)
        # self.bn = nn.BatchNorm3d(256)
        # self.dropout = nn.Dropout(0.15)
        # self.lin2 = nn.Linear(256,125)
        # self.lin3 = nn.Linear(125,25)
        # self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    

    # Not sure exactly how this is supposed to work. Got this from a tutorial
    def forward(self, x):
        
        # print("second initial shape!")
        # print(x.shape)
        # test = x[:,1,:,:,:]
        # print(test.shape)

        # h = x.view(-1,487)
        # h, _ = self.ltsm(h)
        # print("LTSM shape")
        # print(h.shape)
        # h = h.view(-1, 512)
        logits = self.fc1(x)
        probs = self.softmax(logits)
        # logits = logits.view(-1,4,25)

        return probs

net = model_C3D.C3D()
net.load_state_dict(torch.load('c3d.pickle')) # experiment with commenting this out and not
print("BEFORE")
print(net.modules())
net = nn.Sequential(*list(net.modules())[:-1]) # strips off last linear layer
print("AFTER")
print(net)
# net = nn.Sequential(*list(net.modules())[:-1]) # strips off last linear layer

net = nn.Sequential(
        net, classifier()
)

net = nn.DataParallel(net)
net.to(device)
print(net.device_ids)


# if(not torch.cuda.is_available()):
    # print("???")
    # from torchsummary import summary
    # summary(net, (3,16,112,112), batch_size=batch_size)


train_loader, test_loader = readdata.MakeDataloaders() #test_dir="../nvGesture/nvgesture_test_correct_cvpr2016.lst", train_dir="../nvGesture/nvgesture_train_correct_cvpr2016.lst"


def saveModel(acc): 
    ct = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")
    path = "./network_C3D_pretrain1000_" + ct + str(acc) + "_.pth"
    torch.save(net.state_dict(), path) 

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

print("did it make it here?")
# print(net.device)

epochs = 1000

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
        

        # print(device)
        # print(inputs.device)
        # print(labels.device)


        optimizer.zero_grad()
        
        outputs = net(inputs)
        # print("SHAPES")
        # print(outputs.shape)
        # print(outputs)
        # print(labels.shape)
        # print(labels)
        
        
        # log_probs = 8, 10, 26 (HAS BLANK)
        # labels = [10, 8] tensor, each frame has the label eg: [5, 5, 5, 5, 5, 5, 5, 5]
        # input_lengths = torch.tensor(batch_size); - batch size length with all 8s
        # target_lengths = torch.tensor(batch_size); - batch size length with all 8s
        train_loss = criterion(outputs, labels)
        # train_loss = criterion(outputs, labels, input_lengths, target_lengths)
        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()

    train_loss_value = running_train_loss/len(train_loader)

    with torch.no_grad(): 
        net.eval() 
        for data in test_loader:
            inputs, outputs = data 
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                outputs = outputs.cuda()
            predicted_outputs = net(inputs)
            # print(predicted_outputs)
            # print(outputs)
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