import torch
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gesture
from torchvision.transforms import *
import numpy as np
import torch
import torch.nn as nn
import gesture

model = gesture.C3D()
model = nn.Sequential(*list(model.modules())[:-1])
model = nn.Sequential(
    model, gesture.classifier() # could be 8 or 25 classes
)

model.load_state_dict("./network_C3D_pretrain1000_2022-03-07-12-53-20_33.037974683544306_.pth")
model = model.cuda()
model.eval()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS,48)

sequence_length = 8
images = []
value = 0
out = np.zeros(25)

transform  = Compose([transforms.CenterCrop((112,112)), transforms.ToTensor(), transforms.normalize(std=std, mean=mean)])


while(True):

  ret, frame = camera.read()
  resized_frame = cv2.resize(frame, (320,240))
  # image = transform(resized_frame)

  if len(images) < 16:
    images.append(torch.unsqueeze(image, 0))
    # pass

  if len(images) == sequence_length:
    data = torch.cat(images).cuda
    print(data.shape)
    out = model(data.unsqueeze(0))
    print(out)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()