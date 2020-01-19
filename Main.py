from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
import denegement, getvoice
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from threading import Thread
import pyautogui, time, os
import win32com.client as wincl


label_detected = []
counting,counter = 0,0

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

 
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return ((knownWidth * focalLength) / perWidth)  * 2.54 / 96

old_distance,temp_Y1,temp_Y2, temp_X1,temp_X2 = 0 , 0 ,0 ,0, 0


def write(x, img):
    global counting
    global temp_Y1,temp_Y2, temp_X1,temp_X2, old_distance
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    ver = 0
    #get width and height
    x1 = str(c1[0]).split('(')[1].split(',')[0]
    x2 = str(c1[1]).split('(')[1].split(',')[0]
    y1 = str(c2[0]).split('(')[1].split(',')[0]
    y2 = str(c2[1]).split('(')[1].split(',')[0]
    X = int(y1)-int(x1)
    Y = int(y2)-int(x2)
    rotating_1 = int(y2)-int(x1)
    rotating_2 = int(x2)-int(y1)
    surface_old = (temp_Y1-temp_X1)*(temp_Y2-temp_X2)
    #Focal length of the camera
    F = 155* 4.35
    obj_width = 340.15
    obj_height = 1040.15
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    unit = ' cm'
    if X != 0 and ( str(label) == 'traffic light' or str(label) == 'stop sign' or str(label) == 'fire hydrant') :
        if old_distance * 0.9 < int(round(distance_to_camera(obj_height,F,X))) < old_distance * 1.1 and temp_X2 * 0.9 < int(x2) < temp_X2 * 1.1:
            print('standing still ',old_distance)
            counting +=1
            counter +=1
            ver = 1
            if counting == 10 and old_distance <1000:
                TEXT = 'WARNING the distance between you and the '+str(label)+' is '+str(old_distance)+ unit+ 'you should stop immidiatly.'
                Thread(target = text_to_speech(TEXT)).start()
                counting = 0
            if counter == 100:
                denegement.display_ip()
        else:
            temp_ditance = round(distance_to_camera(obj_height,F,X))
            if temp_ditance > 1000:
                unit = ' m'
                temp_ditance = temp_ditance/100
                obj_height = obj_height/100
                print('the distance between the camera and  '+str(label)+' is ',temp_ditance, unit)
                TEXT ='the distance between the camera and  '+str(label)+' is '+str(temp_ditance)+ unit
                counting = 0
            elif temp_ditance < 110:
                print('WARNING the distance between you and the '+str(label)+' is ',temp_ditance, unit, 'you should stop immidiatly.')
                TEXT ='WARNING the distance between you and the '+str(label)+' is '+str(temp_ditance)+ unit+ 'you should stop immidiatly.'
                text_to_speech(TEXT)  
            
        #print(X, '   ' , Y)
        #print(x1, '   ' ,x2, '   ' ,y1, '   ' ,y2)
        if ver == 0 and ((temp_X1 > int(x1) and temp_X2 < int(x2)) or (temp_Y2 < int(y2) and temp_Y1 > int(y1))):
            if surface_old*0.9 < X*Y < surface_old*1.1 :
                print('complex rotation')
            else:
                print('simple rotation ')

        temp_X1,temp_X2 = int(x1), int(x2)
        temp_Y1,temp_Y2 = int(y1), int(y2)
        old_distance = int(round(distance_to_camera(obj_height,F,X)))

    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    tem = cv2.rectangle(img, c1, c2,color, -1)
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    #sizesh =  img.shape
    #height, width = tem.shape[:2]
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

def text_to_speech(TEXT):

    speak = wincl.Dispatch("SAPI.SpVoice")

    if TEXT:
        speak.Speak(TEXT)
        TEXT = ''
    
def main_main():
    global classes, colors
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    

    
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    videofile = 'video.avi'
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            

        
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            #print(float(inp_dim),inp_dim)
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            cv2.imshow("frame", orig_im) #show the frame / output
            key = cv2.waitKey(1)
            #time.sleep(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break

        
def check_voice_comm():
    while True:
        voice = ''
        try:
            voice = getvoice.get_voice()
            print('voice  ',voice)
        except:
            voice = ''
        if 'station' in voice:
            denegement.display_ip()

    
if __name__ == '__main__':
    Thread(target = main_main).start()
    Thread(target = check_voice_comm).start()
    #Thread(target = text_to_speech).start()


    

