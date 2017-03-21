import numpy as np  
import argparse  
import cPickle  
import cv2  
import  glob

args = {"dataset":"D:\ImageProcessing\imagenet\\n01694178",}

for imagePath in glob.glob(args["dataset"] + "/*.JPEG"): 
    k = imagePath[imagePath.rfind("n") + 1:]   
    print k 
    img = cv2.imread(imagePath)
    res=cv2.resize(img,(400,166),interpolation=cv2.INTER_CUBIC)  
    cv2.imwrite("D:\ImageProcPessing\imagenet\ImageData\\"+k, res)
