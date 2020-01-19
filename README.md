# A PyTorch implementation of a YOLO v3 Object Detector

[UPDATE] : This repo serves as a driver code for my research. I just graduated college, and am very busy looking for research internship / fellowship roles before eventually applying for a masters. I won't have the time to look into issues for the time being. Thank you.


This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO). I've also tried to keep the code minimal, and 
document it as well as I can. 

### Tutorial for building this detector from scratch
If you want to understand how to implement this detector by yourself from scratch, then you can go through this very detailed 5-part tutorial series I wrote on Paperspace. Perfect for someone who wants to move from beginner to intermediate pytorch skills. 

[Implement YOLO v3 from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

As of now, the code only contains the detection module, but you should expect the training module soon. :) 

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4
4. Tensorflow
... see the Req. Document
Using PyTorch 0.3 will break the detector.



## Detection Example + Calculating DISTANCE

![Detection Example](https://i.imgur.com/ros7bw3.jpg)



### Speed Accuracy Tradeoff
You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. 

```
python detect.py --images imgs --det det --reso 320
```

### On Video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .avi format
since openCV only accepts OpenCV as the input format. 

```
python video_demo.py --video video.avi
```

Tweakable settings can be seen with -h flag. 


### Get Started
Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, 
feed will be taken from what the OpenCV, recognises as camera 0. The default image resolution is 160 here, though you can change it with `reso` flag.

```
python main.py

or

python maintest.py --video videotest.mp4
```

