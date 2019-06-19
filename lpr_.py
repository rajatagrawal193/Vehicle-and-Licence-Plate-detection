from src.utils import nms
from src.label import dknet_label_conversion
from glob import glob
from os.path import splitext, basename
import sys
import cv2
import numpy as np
import traceback
import os
sys.path.append(os.path.abspath('darknet/python/'))
from darknet import detect
import darknet as dn




# New-branch test commit

class lpr:
    def __init__(self):

        self.ocr_threshold = .4

        self.ocr_weights = 'data/ocr/ocr-net.weights'.encode()
        self.ocr_netcfg = 'data/ocr/ocr-net.cfg'.encode()
        self.ocr_dataset = 'data/ocr/ocr-net.data'.encode()

        self.ocr_net = dn.load_net(self.ocr_netcfg, self.ocr_weights, 0)
        self.ocr_meta = dn.load_meta(self.ocr_dataset)

    # input_dir  = sys.argv[1]
        # self.output_dir = input_dir 

    def plates_ocr(self,img):
        # imgs_paths = sorted(glob('%s/*.png' % self.output_dir))
        # print(imgs_paths)

        #print('Performing OCR...')

        # for i, img_path in enumerate(imgs_paths):
        #     print("Frame {} out of {}".format(i, len(imgs_paths)))
        #     print('\tScanning %s' % img_path)

        # bname=basename(splitext(img_path)[0])
        # print(bname)
        # img= cv2.imread(img_path)
        height, width= img.shape[:2]
        
        R=detect(self.ocr_net, self.ocr_meta,
            img, thresh=self.ocr_threshold, nms=.45)
        #print(R)
        if len(R):

            L=dknet_label_conversion(R, width, height)
            L=nms(L, .45)

            L.sort(key=lambda x: x.tl()[0])
            lp_str=''.join([chr(l.cl()) for l in L])

            # with open('%s/%s_str.txt' % (self.output_dir, bname), 'w') as f:
            #     f.write(lp_str + '\n')

            # print '\t\tLP: %s' % lp_str
            return lp_str

        else:

            # print 'No characters found'
            return 'No characters found'
# if __name__ == "__main__":
    # _lpr=lpr(sys.argv[1])
    # print("here")
    
    # _lpr.plates_ocr()
