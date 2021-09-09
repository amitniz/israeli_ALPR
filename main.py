import cv2 as cv
import numpy as np
import pytesseract




#Best for now
# L:(15,70,90)
# H:(25,255,200)
# E: 0.06
SCALE = 100
LOW_THRESHOLD = (15,70,60)
HIGH_TRESHOLD = (25,255,235)
EPSILON_CONST = 0.06
MIN_WIDTH = 75
MIN_HEIGHT = 15
MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 1500

img_path = 'vid4.mp4'
#img_path = 'car_5.jpeg'
def findLicensePlate(im):
    blured = cv.GaussianBlur(im,(7,7),1)
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    im_hsv = cv.cvtColor(blured,cv.COLOR_BGR2HSV)
    #cv.imshow('blured',blured)
    plate_mask =cv.inRange(im_hsv,LOW_THRESHOLD,HIGH_TRESHOLD)
    
    contours,_ = cv.findContours(plate_mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    cont_rect=[]
    for cont in contours:
        epsilon = EPSILON_CONST*cv.arcLength(cont,closed=True)
        approx = cv.approxPolyDP(cont,epsilon,True)
        if len(approx)==4 and MIN_CONTOUR_AREA < cv.contourArea(approx)<MAX_CONTOUR_AREA:
                cont_rect.append(approx)
            
            
    if len(cont_rect):
        best_match = max(cont_rect,key=cv.contourArea)
        cv.drawContours(im,[best_match],-1,(0,255,0),2)
        x, y, w, h = cv.boundingRect(best_match)
        cropped = gray[y:y+h,x:x+w]
        
        if MIN_WIDTH<= w and MIN_HEIGHT<=h:
            cropped_bw = cv.inRange(cropped,0,50)
            cv.imshow("cropped",cropped_bw)
            text = pytesseract.image_to_string(cropped_bw)
            if sum(c.isdigit() for c in text) == 8:
                print(f"Got Number: {''.join([c if c.isdigit() else '-' for c in text])[:10]}")
            
    return im,blured,plate_mask

cap =cv.VideoCapture(img_path)
ret,im = cap.read()
while ret:
    width = int(im.shape[1] * SCALE / 100)
    height = int(im.shape[0] * SCALE / 100)
    dim = (width, height)
    resized =cv.resize(im,dim, interpolation = cv.INTER_CUBIC)
    img,blured,masked =findLicensePlate(resized)
    #cv.imshow("blur",blured)
    #cv.imshow("mask",masked)
    cv.imshow("plate",img)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
    
    ret,im = cap.read()
    

cap.release()
'''

im = cv.imread(img_path)
img,blured,masked =findLicensePlate(im)
#cv.imshow("blur",blured)
cv.imshow("mask",masked)
cv.imshow("plate",img)
cv.waitKey(0) & 0xFF == ord('q')
'''
cv.destroyAllWindows()
