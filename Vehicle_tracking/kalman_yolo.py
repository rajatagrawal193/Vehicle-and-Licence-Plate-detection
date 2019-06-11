# import the necessary packages
import sys
import os
sys.path.append(os.path.abspath('darknet/python/'))
import numpy as np
import time
import cv2
import darknet as dn
import helpers



class Detector:

    def __init__(self,classes,confidence=0.5):

        self.classes = classes
        
        base_path  =  os.getcwd()
        dn.set_gpu(0)
        self.net = dn.load_net(os.path.join(base_path,"darknet/yolov3.cfg"),os.path.join(base_path,"weights/yolov3.weights"), 0)
        self.meta = dn.load_meta(os.path.join(base_path,"darknet/cfg/coco.data"))
        self.confidence  = confidence
        return

    def get_detected_boxes(self,frame):
    
        boxes2 = []
        currIds=[]
        trackingBoxesData = []
        dic = {}
    
        # if W is None or H is None:
        (H, W) = frame.shape[:2]
    
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        detection_dim = (608,608) 
        confidences = []
        classIDs = []
        start = time.time()
    
        detection_frame  =  cv2.resize(frame,(608,608))
    
        layerOutputs = dn.detect(self.net, self.meta,detection_frame,self.confidence)
        #end = time.time()
        #dt_time  += end-start
        #print("Detection Time:" + str(end-start))
    
            # loop over each of the layer outputs
        for output in layerOutputs:
            # print(output)
            # loop over each of the detections
            label = output[0]
            confidence = output[1]
            box  = np.array(output[2])
            #for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                #scores = detection[5:]
                #classID = np.argmax(scores)
                #confidence = scores[classID]
            if label in self.classes:
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = box * np.array([W/608.0, H/608.0, W/608.0, H/608.0])
                    (centerX, centerY, width, height) = box.astype("int")
    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([y, x, int(height)+y, int(width)+x])
                    confidences.append(float(confidence))
                    #classIDs.append(classID)
                    # if classID:
                    # 	print(classID)
    
        #print(boxes)
        for box in boxes:
    #        threshold=H*W/4000
            threshold  = 0
            if(int(box[2]*box[3])>threshold): 
                boxes2.append(box)
    
    
        return boxes2



