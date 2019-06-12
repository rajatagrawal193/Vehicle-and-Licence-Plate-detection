import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms

#New-branch test commit
class lpr:
    def __init__(self ):
        
        self.ocr_threshold = .4

        self.ocr_weights = 'data/ocr/ocr-net.weights'
        self.ocr_netcfg  = 'data/ocr/ocr-net.cfg'
        self.ocr_dataset = 'data/ocr/ocr-net.data'

        self.ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
        self.ocr_meta = dn.load_meta(ocr_dataset)

        
    input_dir  = sys.argv[1]
    output_dir = input_dir

    

    imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
    print(imgs_paths)

    print 'Performing OCR...'

    for i,img_path in enumerate(imgs_paths):

        print '\tScanning %s' % img_path

        bname = basename(splitext(img_path)[0])
        print(bname)

        R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

        if len(R):

            L = dknet_label_conversion(R,width,height)
            L = nms(L,.45)

            L.sort(key=lambda x: x.tl()[0])
            lp_str = ''.join([chr(l.cl()) for l in L])

            with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                f.write(lp_str + '\n')

            print '\t\tLP: %s' % lp_str

        else:

            print 'No characters found'

	