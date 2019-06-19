
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
import os
import helpers
import tracker as tr
import time
from kalman_yolo import Detector as YoloDetector
import cv2
import sys
import gc
#from lpd_ import lpd
from lpr_ import lpr
from ReID_CNN.Model_Wrapper import ResNet_Loader
from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
from torchvision.ops.boxes import nms
import torch
import argparse




def extract_feature(img,z):

    print(z)
    obj  =  img[z[0]:z[2],z[1]:z[3]]

    features = reid_model.infer(obj).numpy()

    return features

def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = helpers.box_iou2(trk, det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global avg_fps 
    frame_count += 1
    #print("")
    #print(frame_count)
    #print("")
    start = time.time()
    img_dim = (img.shape[1], img.shape[0])

    # YOLO detection for vehicle
    yolo_start = time.time()
    z_box = yolo_det.get_detected_boxes(img)
    #z_box_cpy= z_box
    yolo_end = time.time()

    # Lpd
    
    #print("Time taken for yolo detection is", yolo_end-yolo_start)
    track_start = time.time()
    if debug:
        print('Frame:', frame_count)

    x_box = []
    if debug:
        for i in range(len(z_box)):
            img1 = helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
            cv2.imshow("frame", img1)
            k = cv2.waitKey(10)
            if k == ord('e'):
                cv2.destroyAllWindows()
                sys.exit(-1)

        # plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd=0.3)
    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)

    # Deal with matched detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.features.append(extract_feature(img,z))
            z = np.expand_dims(z, axis=0).T
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            if len(unmatched_trks)>0:
                min_score = 10000000
                tmp_idx = -1
                for trk_idx in unmatched_trks:
                    trk  = tracker_list[trk_idx]
                    #print(len(trk.features))
                    if len(trk.features)==0:
                        continue
                    score  = trk.feature_match(extract_feature(img,z)) ## find closest feature match
                    if score<min_score:
                        min_score= score
                        tmp_idx = trk_idx
                if min_score<feature_thresh and tmp_idx!=-1:
                    
                    z = np.expand_dims(z, axis=0).T
                
                    tmp_trk = tracker_list[tmp_idx]
                    tmp_trk.kalman_filter(z)
                    xx = tmp_trk.x_state.T[0].tolist()
                    xx = [xx[0], xx[2], xx[4], xx[6]]
                    x_box[trk_idx] = xx
                    tmp_trk.box = xx
                    tmp_trk.hits += 1
                    tmp_trk.no_losses = 0
                    continue

            
            #new_boxes.append(z)
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tr.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft()  # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks*100
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx

    # The list of tracks to be annotated
    img_vis  =  img.copy()
    good_tracker_list = []
    #print(img_dim) 
    good_boxes = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            good_boxes.append(trk.box) 
    #for trk in good_tracker_list:
    selected_ids = nms(torch.FloatTensor(np.array(good_boxes)),torch.FloatTensor([1.0]*len(good_boxes)),0.45)
    for idx in selected_ids:
        trk = good_tracker_list[idx]
        x_cv2 = trk.box
        idx = trk.id
        if debug:
            print('updated box: ', x_cv2)
            print()
        # Draw the bounding boxes on the
        img_vis = helpers.draw_box_label(img_vis, x_cv2, idx)
        if frame_count%5==0:
            y1_temp, x1_temp, y2_temp, x2_temp= x_cv2 
            
            w_temp= x2_temp-x1_temp

            h_temp= y2_temp- y1_temp
            if w_temp*h_temp <400 or w_temp<=0 or h_temp<=0 or min(x_cv2)<0:

                continue
            plates = []
            #print(x_cv2)
            dt_start = time.time()
            Ivehicle = img[y1_temp:y2_temp, x1_temp:x2_temp]
            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side = int(ratio*288.)
            bname  = 'frame{}_{}.png'.format(frame_count,idx) 

            bound_dim = min(side + (side % (2**4)), size)
            # print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)
            #dt_plates_start = time.time()
            Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)
            if len(LlpImgs):

                plates = [Llp[0].pts]
                cv2.imwrite("%s/%s" %(detected_plates_dir, bname),LlpImgs[0]*255.)
                plate_string= _lpr.plates_ocr(LlpImgs[0]*255.)
                for plate in plates:
                    x1 = (plate[0][0]*w_temp +x1_temp).astype('int')
                    y1 = (plate[1][0]*h_temp +y1_temp).astype('int')
                    x2 = (plate[0][1]*w_temp +x1_temp).astype('int')
                    y2 = (plate[1][1]*h_temp +y1_temp).astype('int')
                    x3 = (plate[0][2]*w_temp +x1_temp).astype('int')
                    y3 = (plate[1][2]*h_temp +y1_temp).astype('int')
                    x4 = (plate[0][3]*w_temp +x1_temp).astype('int')
                    y4 = (plate[1][3]*h_temp +y1_temp).astype('int')
                    
                    plate = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    plate = plate.reshape((-1, 1, 2))
                    cv2.polylines(img_vis, [plate], True, (255, 0, 0),4)
                    cv2.putText(img_vis, plate_string,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1.1,(0,0,255),2)
                cv2.imwrite("%s/%s" %(detected_cars_dir, bname),img_vis[y1_temp:y2_temp,x1_temp:x2_temp])
    track_end = time.time()

                                   # images
#    dt_start = time.time()


    print("Time taken to track the boxes is", track_end-track_start)
    end = time.time()
    fps = 1.0/(end-start)
    #dt_fps = 1.0/(dt_dr+yolo_end-yolo_start)
    avg_fps += fps
    cv2.putText(img_vis, "FPS: {:.4f}".format(fps), (int(
        0.8*img_dim[0]), 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 4)
    #cv2.putText(img_vis, "Detect FPS: {:.4f}".format(
    #    dt_fps), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255, 0), 4)
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > feature_tp, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= feature_tp]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))

    return img_vis


if __name__ == "__main__":

    #det = detector.C5rDe0tector(2
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('--output', default = 'output', help='Path to output video file')
    parser.add_argument('--size', default=416, type=int, help='set the size of the input frame for yolo. Supported sizes are 416 and 608')
    parser.add_argument('--f_th', default=550, type=float, help='feature distance matching threshold')
    parser.add_argument('--max_age', default=10, type=int, help='Frames for which kalman should predict if vehicle detection is lost. Features are stored for reidentification for max_age*scale')
    parser.add_argument('--f_tp', default=2000, type=int, help='Frames for which features are stored for reidentification. Increase this number if camera covers a larger scene')

    args =  parser.parse_args()
    size  = args.size
    feature_thresh = args.f_th
    max_age = args.max_age  #no.of consecutive unmatched detection before a track is stopped being tracked
    feature_tp = args.f_tp

    # Global variables to be used by funcitons of VideoFileClop
    frame_count = 0  # frame counter
    
    min_hits = 1  # no. of consecutive matches needed to establish a track
    
    tracker_list = []  # list for trackers
    # list for track ID
    unq_ids = ['id' + str(i) for i in range(100)]
    track_id_list = deque(unq_ids)
    #track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L','M','N','O'])
    avg_fps  = 0

    debug = False
    interestedClass = ["car", "truck", "bus", "train"] # Interested classes for yolo
    #Initialize yolo detector
    yolo_det = YoloDetector(interestedClass,size=size)
    #Initialize ReID model
    reid_model_path  =  os.path.abspath('ReID_CNN/model_880_base.ckpt')
    reid_model = ResNet_Loader(reid_model_path,50, 32)
    #Intialize License Plate detector
    lp_model ="data/lp-detector/wpod-net_update1.h5"
    wpod_net = load_model(lp_model)
    #License Plate detection threshold
    lp_threshold=0.8
    
    # test on a video file.
    input_file = args.input
    output_dir  = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_basename  = os.path.basename(input_file)    
    detected_plates_dir  = output_dir + '/' + 'license_plates'
    detected_cars_dir  = output_dir + '/' + 'cars'
    if not os.path.exists(detected_plates_dir):
        os.makedirs(detected_plates_dir)
    if not os.path.exists(detected_cars_dir):
        os.makedirs(detected_cars_dir)

    output = output_dir + '/' +input_basename.split('.mp4')[0] + '_result_tracker'  + str(size) + '.avi'
    start = time.time()
    #lpd = lpd("data/lp-detector/wpod-net_update1.h5")
    cap = cv2.VideoCapture(input_file)
    #Intitialize OCR for LPD
    _lpr= lpr()
    success = True
    while success:
        success, img = cap.read()
        
        if frame_count==0:
            frame_height,frame_width = img.shape[:2]
            out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
        if success:
            start_pipeline = time.time()
            img = pipeline(img)
            end_pipeline = time.time()
            print("Time taken for this iteration is",
                  round(end_pipeline-start_pipeline, 2))
            #cv2.imshow("frame", img)
            out.write(img)
            #cv2.waitKey(1)
    cap.release()
    end = time.time()

    print(round(end-start, 2), 'Seconds to finish')
    print(avg_fps/(frame_count*1.0), 'Average FPS')
