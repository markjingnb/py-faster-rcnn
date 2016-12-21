#!/usr/bin/env python
# -*- coding: utf-8 -*
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import scipy.io as sio
caffe_root = '/home/ubuntu/jnb/py-faster-rcnn/caffe-fast-rcnn/'  
import sys  
sys.path.insert(0, caffe_root + 'python')  
import caffe  
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

import os, sys, cv2
import argparse
myfont = mplt.font_manager.FontProperties(fname='/usr/share/fonts/truetype/msyh.ttf')

CLASSES = ('__background__',
           u"飞机", u"自行车", u"小鸟", u"船",
           u"瓶子", u"巴士", u"汽车", u"猫", u"椅子",
           u"牛", u"餐桌", u"狗", u"马",
           u"摩托车", u"人", u"盆栽",
           u"羊", u"沙发", u"火车", u"显示器")

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    print'jnb1\n\n\n'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    #fig, ax = plt.plot(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#print('111111111111111111111\n\n\n')
        #vis_detections(im, cls, dets, thresh=CONF_THRESH,)
#        print('jnjnjnjjnjnjjnjjjjjjjjjj\n\n\n')
#####################################################
        class_name=cls
        inds = np.where(dets[:, -1] >= 0.8)[0]
#        print('begin\n\n\n')
        if len(inds) == 0:
            continue
#        print('123\n\n\n')
        #im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        #ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
            	plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            			)
            ax.text(bbox[0], bbox[1] - 2,
                	#'{:s} {:.3f}'.format(class_name, score),
		class_name,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white',fontproperties=myfont)

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
#        print('end\n\n\n')
    #ax.imwrite('123.jpg')
    plt.savefig('123.jpg', dpi = 400, bbox_inches = "tight")  

    timer.toc()
    print ('draw time {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #plt.show()
    im1 = cv2.imread('123.jpg')
    return im1

###############################################
def demo_img(net, image):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)
    im=image

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig2, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#print('111111111111111111111\n\n\n')
        #vis_detections(im, cls, dets, thresh=CONF_THRESH,)
#        print('jnjnjnjjnjnjjnjjjjjjjjjj\n\n\n')
#####################################################
        class_name=cls
        inds = np.where(dets[:, -1] >= 0.8)[0]
#        print('begin\n\n\n')
        if len(inds) == 0:
            continue
#        print('123\n\n\n')
        #im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        #ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
            	plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            			)
            ax.text(bbox[0], bbox[1] - 2,
                	#'{:s} {:.3f}'.format(class_name, score),
		class_name,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white',fontproperties=myfont)

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
#        print('end\n\n\n')
    #ax.imwrite('123.jpg')
    plt.savefig('123.jpg', dpi = 400, bbox_inches = "tight")  
    plt.close(fig2)#very  important
    timer.toc()
    print ('draw time {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    im1 = cv2.imread('123.jpg')
    return im1

###############################################

def jnb_demo_video(net):
	#fig, ax = plt.subplots(figsize=(12, 12))
	fig, ax = plt.subplots()
	plt.axis('off')
	#im1 = cv2.imread('jnb.jpg')
	#im1 = im1[:, :, (2, 1, 0)]
	#ax.imshow(im1, aspect='equal')
	capture = cv2.VideoCapture('1.MOV')
	#fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
	#size = (int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), 
	#                int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
	#print fps,size

	ims = []

	for i in range(200):
	#im = plt.imshow(im1, animated=True)
		success, im2 = capture.read()
		im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
		im2=demo_img(net, im2)
		im = plt.imshow(im2, animated=True)    
		ims.append([im])
	ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True)
	FFwriter = animation.FFMpegWriter()	
        #plt.show()
	#ani.save('dynamic_images.mp4',writer = FFwriter,fps=2, extra_args=['-vcodec', 'libx264'])
	ani.save('dynamic_images.mp4',dpi=500)
	plt.show()
##ffmpeg -i dynamic_images.mp4 -vf crop=iw-2*650:ih-2*700:650:700 p7t.mp4
############################################### 


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    #if args.cpu_mode:
    #    caffe.set_mode_cpu()
    #else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

#    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
 #               '001763.jpg', '004545.jpg']
    im_names = ['jnb.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        #demo(net, im_name)
        jnb_demo_video(net)

    #plt.show()
