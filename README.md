# Drone Control with Gesture Recognition

This repository has a number of scripts and files used for deep learning for gesture control. There were three manners of control investigated:

* 3D CNNs for dynamic gestures, supported by the scripts in /nvGesture and /Gesture_Recognition. These are intended to be run on the nvGesture dataset, and contain two networks: an implimentation of C3D and an implimentation of Resnet2_plus1.

* Parallelized 1D CNNs for dynamic gestures using XYZ points from skeletal hand data, as collected by MediaPipe. Pretrained weights for this model are also included.

* 2D CNNs for static gesture recognition in a video stream using depth or infrared data. This was used to control the drone in the demonstration of the project.

https://youtu.be/BBnZKu9_Neg

This is further explained on the portfolio post https://javtges.github.io/gesturerecognition/