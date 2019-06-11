import sys
import os
import keras
import cv2
import traceback
import time

from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes


def adjust_pts(pts, lroi):
    return pts*lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


class lpd:
    def __init__(self, lp_model):
        self.wpod_net = load_model(lp_model)

    def detectPlates(self,img, box, frame_count, i ):
        print("inside-1")

        output_dir = 'output'
        detected_cars_dir = "detected_cars"
        detected_plates_dir = "detected_plates"
        lp_threshold = .5
        bname = 'frame{}_{}.png'.format(frame_count, i)
        

        plates = []

        if min(box) > 0:
        # print(box)
            print("inside-2")

            Ivehicle = img[box[0]:box[2], box[1]:box[3]]
            cv2.imwrite("%s/%s/%s" %
                        (output_dir, detected_cars_dir, bname), Ivehicle)
            # print(Ivehicle.shape[:2])
            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side = int(ratio*288.)
            bound_dim = min(side + (side % (2**4)), 608)
            # print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)
            dt_plates_start = time.time()
            Llp, LlpImgs, _ = detect_lp(self.wpod_net, im2single(
            Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)
            dt_plates_end = time.time()
            print("Time taken for for internal plate detection",
            dt_plates_end-dt_plates_start)

            if len(LlpImgs):
                print("inside-3")
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)
                # print(Llp[0].pts)
                plates.append(Llp[0].pts)
                # cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
                cv2.imwrite("%s/%s/%s" %
                        (output_dir, detected_plates_dir, bname), Ilp*255.)

            # writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
        return plates
    # except:
    # 	traceback.print_exc()
    # 	sys.exit(1)

    # sys.exit(0)
