#!/usr/bin/env python
# -*- coding: utf-8 -*
import sys
sys.path.insert(0,'../caffe/python')

import caffe
import numpy as np
from matplotlib import pylab as pl
import matplotlib as mplt
#import pylab as pl
import os
import cv2
import time
import anno_func, json
#%matplotlib inline
gid = 0

myfont = mplt.font_manager.FontProperties(fname='/usr/share/fonts/truetype/msyh.ttf')
#cls=[u'i1', u'i10', u'i11', u'i12', u'i13', u'i14', u'i15', u'i2', u'i3', u'i4', u'i5', u'il100', u'il110', u'il50', u'il60', u'il70', u'il80', u'il90', u'io', u'ip', u'p1', u'p10', u'p11', u'p12', u'p13', u'p14', u'p15', u'p16', u'p17', u'p18', u'p19', u'p2', u'p20', u'p21', u'p22', u'p23', u'p24', u'p25', u'p26', u'p27', u'p28', u'p3', u'p4', u'p5', u'p6', u'p7', u'p8', u'p9', u'pa10', u'pa12', u'pa13', u'pa14', u'pa8', u'pb', u'pc', u'pg', u'ph1.5', u'ph2', u'ph2.1', u'ph2.2', u'ph2.4', u'ph2.5', u'ph2.8', u'ph2.9', u'ph3', u'ph3.2', u'ph3.5', u'ph3.8', u'ph4', u'ph4.2', u'ph4.3', u'ph4.5', u'ph4.8', u'ph5', u'ph5.3', u'ph5.5', u'pl10', u'pl100', u'pl110', u'pl120', u'pl15', u'pl20', u'pl25', u'pl30', u'pl35', u'pl40', u'pl5', u'pl50', u'pl60', u'pl65', u'pl70', u'pl80', u'pl90', u'pm10', u'pm13', u'pm15', u'pm1.5', u'pm2', u'pm20', u'pm25', u'pm30', u'pm35', u'pm40', u'pm46', u'pm5', u'pm50', u'pm55', u'pm8', u'pn', u'pne', u'po', u'pr10', u'pr100', u'pr20', u'pr30', u'pr40', u'pr45', u'pr50', u'pr60', u'pr70', u'pr80', u'ps', u'pw2', u'pw2.5', u'pw3', u'pw3.2', u'pw3.5', u'pw4', u'pw4.2', u'pw4.5', u'w1', u'w10', u'w12', u'w13', u'w16', u'w18', u'w20', u'w21', u'w22', u'w24', u'w28', u'w3', u'w30', u'w31', u'w32', u'w34', u'w35', u'w37', u'w38', u'w41', u'w42', u'w43', u'w44', u'w45', u'w46', u'w47', u'w48', u'w49', u'w5', u'w50', u'w55', u'w56', u'w57', u'w58', u'w59', u'w60', u'w62', u'w63', u'w66', u'w8', u'wo', u'i6', u'i7', u'i8', u'i9', u'ilx', u'p29', u'w29', u'w33', u'w36', u'w39', u'w4', u'w40', u'w51', u'w52', u'w53', u'w54', u'w6', u'w61', u'w64', u'w65', u'w67', u'w7', u'w9', u'pax', u'pd', u'pe', u'phx', u'plx', u'pmx', u'pnl', u'prx', u'pwx', u'w11', u'w14', u'w15', u'w17', u'w19', u'w2', u'w23', u'w25', u'w26', u'w27', u'pl0', u'pl4', u'pl3', u'pm2.5', u'ph4.4', u'pn40', u'ph3.3', u'ph2.6']

cls=[u'i1', u'i10', u'i11', u'i12', u'i13', u'i14', u'i15', u'i2', u'i3', u'i4', u'i5', u'il100', u'il110', u'il50', u'il60', u'il70', u'il80', u'il90', u'io', u'ip', u'p1', u'p10', u'p11', u'英国', u'p13', u'p14', u'p15', u'p16', u'p17', u'p18', u'p19', u'p2', u'p20', u'p21', u'p22', u'p23', u'p24', u'p25', u'p26', u'p27', u'p28', u'p3', u'p4', u'p5', u'p6', u'p7', u'p8', u'p9', u'pa10', u'pa12', u'pa13', u'pa14', u'pa8', u'pb', u'pc', u'pg', u'ph1.5', u'ph2', u'ph2.1', u'ph2.2', u'ph2.4', u'ph2.5', u'ph2.8', u'ph2.9', u'ph3', u'ph3.2', u'ph3.5', u'ph3.8', u'ph4', u'ph4.2', u'ph4.3', u'ph4.5', u'ph4.8', u'ph5', u'ph5.3', u'ph5.5', u'pl10', u'pl100', u'pl110', u'pl120', u'pl15', u'pl20', u'pl25', u'pl30', u'pl35', u'pl40', u'pl5', u'pl50', u'pl60', u'pl65', u'pl70', u'pl80', u'pl90', u'pm10', u'pm13', u'pm15', u'pm1.5', u'pm2', u'pm20', u'pm25', u'pm30', u'pm35', u'pm40', u'pm46', u'pm5', u'pm50', u'pm55', u'pm8', u'pn', u'pne', u'po', u'pr10', u'pr100', u'pr20', u'pr30', u'pr40', u'pr45', u'pr50', u'pr60', u'pr70', u'pr80', u'ps', u'pw2', u'pw2.5', u'pw3', u'pw3.2', u'pw3.5', u'pw4', u'pw4.2', u'pw4.5', u'w1', u'w10', u'w12', u'w13', u'w16', u'w18', u'w20', u'w21', u'w22', u'w24', u'w28', u'w3', u'w30', u'w31', u'w32', u'w34', u'w35', u'w37', u'w38', u'w41', u'w42', u'w43', u'w44', u'w45', u'w46', u'w47', u'w48', u'w49', u'w5', u'w50', u'w55', u'w56', u'w57', u'w58', u'w59', u'w60', u'w62', u'w63', u'w66', u'w8', u'wo', u'i6', u'i7', u'i8', u'i9', u'ilx', u'p29', u'w29', u'w33', u'w36', u'w39', u'w4', u'w40', u'w51', u'w52', u'w53', u'w54', u'w6', u'w61', u'w64', u'w65', u'w67', u'w7', u'w9', u'pax', u'pd', u'pe', u'phx', u'plx', u'pmx', u'pnl', u'prx', u'pwx', u'w11', u'w14', u'w15', u'w17', u'w19', u'w2', u'w23', u'w25', u'w26', u'w27', u'pl0', u'pl4', u'pl3', u'pm2.5', u'ph4.4', u'pn40', u'ph3.3', u'ph2.6']

cls2=[u'禁鸣',u'禁停']

datadir = "../../data/"

filedir = datadir + "/annotations.json"
ids = open(datadir + "/test/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())

netname = "ours"
model_file = '../model/model.prototxt'
model_weights = "../model/model.caffemodel"
gid = -1


def get_net():
    global gid
    if gid != -1:
        caffe.set_mode_gpu()
        caffe.set_device(gid)
        print "Using GPU id", gid
    else:
        caffe.set_mode_cpu()
        print "Using CPU"
    net = caffe.Net(model_file, 
                 model_weights, caffe.TEST)
    return net

net = get_net()
imgdata = pl.imread('/home/ubuntu/jnb/TT100K/data/test2/m2.jpg')
if imgdata.max() > 2:
        imgdata = imgdata/255.
imgdata = imgdata[:,:,[2,1,0]]*255.# - mn
rimgdata = (imgdata-imgdata.min())/(imgdata.max()-imgdata.min())
#pl.imshow(rimgdata[:,:,[2,1,0]])
#pl.show()
#exit()



def draw_rects(image, rects, color=(1,0,0), width=2):
    if len(rects) == 0: return image
    for i in range(rects.shape[0]):
        xmin, ymin, w, h = rects[i, :].astype(np.int)
        xmax = xmin + w
        ymax = ymin + h
        image[ymin:ymax+1, xmin:xmin+width, :] = color
        image[ymin:ymax+1, xmax:xmax+width, :] = color
        image[ymin:ymin+width, xmin:xmax+1, :] = color
        image[ymax:ymax+width, xmin:xmax+1, :] = color 
    return image

def fix_box(bb, mask, xsize, ysize, res):
    bb = np.copy(bb)
    
    y_offset = np.array([np.arange(0, ysize, res)]).T
    y_offset = np.tile(y_offset, (1, xsize/res))
    x_offset = np.arange(0, xsize, res)
    x_offset = np.tile(x_offset, (ysize/res, 1))
    bb[0, :, :] += x_offset
    bb[2, :, :] += x_offset
    bb[1, :, :] += y_offset
    bb[3, :, :] += y_offset
    
    mask = np.array([mask]*4)
    sb = bb[mask].reshape((4,-1))
    
    rects = sb.T

    rects = rects[np.logical_and((rects[:, 2] - rects[:, 0]) > 0, (rects[:, 3] - rects[:, 1]) > 0), :]
    rects[:, (2, 3)] -= rects[:, (0, 1)]
    
    return rects

def work(imgdata, all_rect,name):
    data_layer = net.blobs['data']
    #for resize in [0.5,1,2,4]:
    for resize in [0.5]:
        prob_th = 0.95
        gbox = 0.1
        if resize < 1:
            resize = data_layer.shape[2]*1.0/imgdata.shape[0]
            data = cv2.resize(imgdata, (data_layer.shape[2], data_layer.shape[3]))
        else:
            data = cv2.resize(imgdata, (imgdata.shape[0]*resize, imgdata.shape[1]*resize))
        data = data.transpose(2,0,1)
        print data.shape
        #data_layer.reshape(*((1,)+data.shape))
        netsize = 1024
        overlap_size = 256

        res1 = 4
        res2 = 16
        pixel_whole = np.zeros((1,data.shape[1]/res1,data.shape[2]/res1))
        bbox_whole = np.zeros((4,data.shape[1]/res1,data.shape[2]/res1))
        type_whole = np.zeros((1,data.shape[1]/res2,data.shape[2]/res2))

        tmp = 0
        for x in range((data.shape[1]-1)/netsize+1):
            xl = min(x*netsize, data.shape[1]-netsize-overlap_size)
            xr = xl+netsize+overlap_size
            xsl = xl if xl==0 else xl+overlap_size/2
            xsr = xr if xr==data.shape[1] else xr-overlap_size/2
            xtl = xsl - xl
            xtr = xsr - xl
            for y in range((data.shape[2]-1)/netsize+1):
                yl = min(y*netsize, data.shape[2]-netsize-overlap_size)
                yr = yl+netsize+overlap_size
                ysl = yl if yl==0 else yl+overlap_size/2
                ysr = yr if yr==data.shape[2] else yr-overlap_size/2
                ytl = ysl - yl
                ytr = ysr - yl
                #print xl,xr,yl,yr,xsl,xsr,ysl,ysr,xtl,xtr,ytl,ytr
                fdata = data[:,xl:xr,yl:yr]


                data_layer.data[...] = fdata
                net.forward()
                pixel = net.blobs['output_pixel'].data[0]
                pixel = np.exp(pixel) / (np.exp(pixel[0]) + np.exp(pixel[1]))
                bbox = net.blobs['output_bb'].data[0]
                mtypes = net.blobs['output_type'].data[0]
                mtypes = np.argmax(mtypes, axis=0)
                print pixel.shape, bbox.shape, mtypes.shape, pixel[1,xtl/res1:xtr/res1, ytl/res1:ytr/res1].shape

                pixel_whole[:,xsl/res1:xsr/res1,ysl/res1:ysr/res1] = pixel[1,xtl/res1:xtr/res1, ytl/res1:ytr/res1]
                bbox_whole[:,xsl/res1:xsr/res1,ysl/res1:ysr/res1] = bbox[:,xtl/res1:xtr/res1, ytl/res1:ytr/res1]
                type_whole[:,xsl/res2:xsr/res2,ysl/res2:ysr/res2] = mtypes[xtl/res2:xtr/res2, ytl/res2:ytr/res2]
                if resize<1: break
            if resize<1: break

        pl.imsave('pixel/'+name+'.png',pixel_whole[0])
	#pl.imshow(pixel_whole[0])
        #pl.show()

        pl.imsave('type/'+name+'.png',type_whole[0])
        #pl.imshow(pixel_whole[0])
	#pl.show()        
        #pl.imshow(bbox_whole)
        #pl.show()        

        rects = fix_box(bbox_whole, pixel_whole[0]>prob_th, imgdata.shape[0]*resize, imgdata.shape[1]*resize, res1)
        merge_rects, scores = cv2.groupRectangles(rects.tolist(), 2, gbox)
        merge_rects = np.array(merge_rects, np.float32) / resize
        imgdraw = imgdata.copy()
	imgdraw = (imgdraw-imgdraw.min())/(imgdraw.max()-imgdraw.min())
        imgdraw2=draw_rects(imgdraw, merge_rects)
        print len(merge_rects)
        #pl.figure(figsize=(20,20))
        #pl.imshow(imgdraw2)
        #pl.show
        #pl.imshow(imgdraw2[:,:,[2,1,0]])
        #pl.show()  
        
        
        mrect = merge_rects * resize / res2 
        if len(mrect)>0:
            mrect[:,[2,3]]+=mrect[:,[0,1]]

        for i,rect in enumerate(mrect):
            xl = np.floor(rect[0])
            yl = np.floor(rect[1])
            xr = np.ceil(rect[2])+1
            yr = np.ceil(rect[3])+1
            xl = np.clip(xl, 0, type_whole.shape[1])
            yl = np.clip(yl, 0, type_whole.shape[2])
            xr = np.clip(xr, 0, type_whole.shape[1])
            yr = np.clip(yr, 0, type_whole.shape[2])

            tp = type_whole[0,yl:yr,xl:xr]
            uni, num = np.unique(tp, return_counts=True)
            maxtp, maxc = 0,0
            for tid, c in zip(uni, num):
                if tid != 0 and maxc<c:
                    maxtp, maxc = tid, c
            if maxtp != 0:
                all_rect.append((int(maxtp), merge_rects[i].tolist(), float(scores[i]), resize))
                #print maxtp, maxc, annos['types'][int(maxtp-1)]

        #pl.figure(figsize=(20,20))               
        #for i,rect in enumerate(all_rect):
	#    pl.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+10, cls[int(np.ceil(rect[0]))], fontsize=20,color='red',fontproperties=myfont)

        #pl.imshow(imgdraw2[:,:,[2,1,0]])
        #pl.imsave('out/'+name+'.png',imgdraw2[:,:,[2,1,0]])
	#pl.imshow(imgdraw2[:,:,[2,1,0]])
        #pl.show()        


        fig2, ax = pl.subplots(figsize=(9, 9),frameon=False)
        ax = pl.Axes(fig2, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig2.add_axes(ax)
        ax.imshow(imgdraw2[:,:,[2,1,0]],aspect='auto')
        #ax.text(600, 600, '4', fontsize=20,color='red',fontproperties=myfont)
        #ax.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+10, cls2[0], fontsize=20,color='red',fontproperties=myfont)
        #ax.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+200, cls2[1], fontsize=20,color='red',fontproperties=myfont)
	for i,rect in enumerate(all_rect):
	    if i==0:
	    	ax.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+10, cls2[0], fontsize=20,color='red',fontproperties=myfont)
	    if i==1:
	    	ax.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+10+50, cls2[1], fontsize=20,color='red',fontproperties=myfont)

	#    	ax.text(np.ceil(rect[1])[0]+10, np.ceil(rect[1])[1]+10, cls2[i], fontsize=20,color='red',fontproperties=myfont)
	extent = ax.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        pl.savefig('out/'+name+'.png', dpi = 120,bbox_inches=extent)
        pl.close(fig2)


#######################################300-55
for i in range(1):
	name=str(30+i)
	imgdata = pl.imread('/home/ubuntu/jnb/TT100K/data/test2/video/'+name+'.png')
	if imgdata.max() > 2:
        	imgdata = imgdata/255.
	imgdata = imgdata[:,:,[2,1,0]]*255.
	all_rect = []
	print i,name
	work(imgdata, all_rect,name)
#imgdraw2 = (imgdata-imgdata.min())/(imgdata.max()-imgdata.min())  
exit()
####################################################################

###################
capture = cv2.VideoCapture('/home/ubuntu/jnb/TT100K/data/test2/IMG_3648.MP4')
for i in range(0):
	success, im2 = capture.read()

for i in range(300):
	success, im2 = capture.read()
	im3=im2[0:1080,0:1080]
	cv2.imwrite('/home/ubuntu/jnb/TT100K/data/test2/video/'+str(i)+'.png',im3);
	print(i)
##ffmpeg -i dynamic_images.mp4 -vf crop=iw-2*650:ih-2*700:650:700 p7t.mp4
exit()
##########line 6













