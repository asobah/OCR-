import sys
import numpy as np
import cv2
from numpy.linalg import norm

letters =   u'\u0780\u0781\u0782\u0783\u0784\u0785\u0786\u0787\u0788\u0789' \
            u'\u078A\u078B\u078C\u078D\u078E\u078F\u0790\u0791\u0792\u0793\u0794\u0795\u0796\u0797'
            
vowel   =   u'\u07A6\u07A7\u07AA\u07AB\u07AC\u07AD\u07AE\u07AF\u07B0'


def pre_process(img):
    ret,des = cv2.threshold(img,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return des


def split_lines(img):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dilation_img = cv2.dilate(img,rect_kernel,iterations = 4)

    y = np.sum(dilation_img, axis=1)

    lines = []
    start = 0
    line_start =0

    for i in range(len(y)):
        if y[i] == 0  and start == 1:
            lines.append(img[line_start:i,])
            start =0
        if y[i] > 0 and start == 0:
            start =1
            line_start = i
    return lines

def split_words(img):

    words = []
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    dilation_img = cv2.dilate(img,rect_kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(dilation_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for h,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        words.append(img[:,x:x+w])

    return words

def split_chars(img):

    chars = []
    contor_image = img.copy()
    sorce_color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(contor_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(sorce_color, contours, -1, (0,255,0), 1)

    for h,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        mask = mask[y:y+h,x:x+w]
        fixed_size = cv2.resize(mask, (20, 20)) 
        ret,fixed_size = cv2.threshold(fixed_size,0, 255, cv2.THRESH_OTSU)
        chars.append([cv2.boundingRect(cnt),fixed_size])

    chars.sort()
    return chars


# start of app
import sys


img = cv2.imread(sys.argv[1],0)
#img = cv2.imread('d:/word.png',0)
pro_img = pre_process(img)

lines = split_lines(pro_img)

#cv2.imshow('preprocess',lines[1])


for l in lines:
     words = split_words(l)
     for w in words:
         chars = split_chars(w)
         for c in chars:
             cv2.imshow('1',lines[0])
             

cv2.waitKey(0)
cv2.destroyAllWindows()