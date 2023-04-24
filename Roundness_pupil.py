import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#pupil detection algorithm 


alpha = 0.8 # Contrast control,optimal for my surroundings
beta = 20 # Brightness control


er_ker = np.ones((20,20),np.uint8)
di_ker = np.ones((15,15),np.uint8)


# return radius,convexity and roundness of all the contours at particular threshold value
def pupil(ip):
    rou_ar = []
    con_ar = []
    radius_ar = []
    for i in range(1,len(ip)):
        area = cv2.contourArea(ip[i])
        peri = cv2.arcLength(ip[i],True)
        hull = cv2.convexHull(ip[i])
        area_hull = cv2.contourArea(hull)
        rou = (4*np.pi*area)/(peri*peri) #roundness
        con = area/area_hull  #convexity
        (x,y),radius = cv2.minEnclosingCircle(ip[i])
        radius = int(radius)   #approx radius of contour
        rou_ar.append(rou)
        con_ar.append(con)
        radius_ar.append(radius)

    return rou_ar,con_ar,radius_ar


# Slelects which out of many countours is to be choosen
def contour(mat,mat2):
    final = []
    for i in range(len(mat)):
        final.append(mat2[mat[i]])
    if  len(mat)==1:
        return mat[final.index(np.max(final))]+1
    else:
        return mat[final.index(np.max(final))]


# Process Input Image
def process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    bf_img = cv2.bilateralFilter(adjusted,25,75,75)
    erode_img = cv2.erode(bf_img,er_ker,iterations=1)
    di_img = cv2.dilate(erode_img,di_ker,iterations=1)
    return di_img

# selects an appropriate threshold value for image
def thres_select(image):
    im_1 = process(image)
    a = 30  #change these values according to the environment
    b = 60
    while(True):
        rani = random.randint(a,b)
        _ , thresh = cv2.threshold(im_1,rani,255,cv2.THRESH_BINARY)
        cont , _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _,_,c = pupil(cont)
        if len(c) == 0:
            a = rani
        elif np.max(c) > 100:
            b = rani
        elif np.max(c) < 15:
            continue
        else:
            print("start",a,"stop=",b)
            break
            
    return a,b


# print(start, " " , stop)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if (len(faces) == 0):
        continue     

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        if (len(eyes) == 0):
            continue
        for (ex, ey, ew, eh) in eyes:
            ey=ey+int(eh/4)
            eh=int(eh/2)
            cv2.rectangle(roi_color, (ex, ey),
                            (ex + ew, ey + eh), (0, 255, 0), 2)
            
            frame1 = frame[faces[0][1]:faces[0][1]+faces[0]
                            [3], faces[0][0]:faces[0][0]+faces[0][2]]

            eye_orig_image = frame1[eyes[0][1]:(
                eyes[0][1]+eyes[0][3]), eyes[0][0]:(eyes[0][0]+eyes[0][2])]

            cv2.imshow('iris', eye_orig_image)
            scale_percent = 500 # percent of original size
            width = int(eye_orig_image.shape[1] * scale_percent / 100)
            height = int(eye_orig_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            eye_orig_image = cv2.resize(eye_orig_image, dim, interpolation = cv2.INTER_AREA)
            
            start,stop =thres_select(eye_orig_image)
            print(start,stop)
           

            for i in range(start,stop):
                img=eye_orig_image
                dil_img = process(eye_orig_image)
                _ , thresh = cv2.threshold(dil_img,i,255,cv2.THRESH_BINARY)
                _ , thresh1 = cv2.threshold(dil_img,i+1,255,cv2.THRESH_BINARY)
                cont , _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cont=cont[1:]   
                cont1 , _ = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cont1=cont1[1:]
                a,_,c = pupil(cont)
                _,_,c1 = pupil(cont1)
                a = np.array(a)
                if len(a)!=0 and len(c1)!=0:
                    if np.max(c1) > 50 and np.max(c1) < 100:
                        mat = np.where(a > 0.6)[0]
                        if(len(mat)!=0):
                            # M = cont(contour(mat,c))
                            # x = int(M['m10']/M['m00'])
                            # y = int(M['m01']/M['m00'])
                            # # print("reaching area more than 50")
                            # # print("iteration=",i)
                            # cv2.circle(img, (x,y), 1,(255,0,0), 1)  
                            # cv2.imshow('pupil',img)         
                            cv2.drawContours(img, cont, contour(mat,c), (0,255,0), 3)
                            cv2.imshow("frame",img)
                            # cv2.waitKey(0)
                            break    
                     
                    elif np.max(c1) - np.max(c) >10:   # checking merger
                        mat = np.where(a > 0.6)[0]
                        if(len(mat!=0)):
                            # M = cont(contour(mat,c))
                            # x = int(M['m10']/M['m00'])
                            # y = int(M['m01']/M['m00'])
                            # cv2.circle(img, (x,y), 1,(255,0,0), 1)
                            # cv2.imshow('pupil',img)
                            # # print("iteration=",i)
                            cv2.drawContours(img, cont, contour(mat,c), (0,255,0), 3)
                            cv2.imshow("frame",img)
                            # cv2.waitKey(30)
                            break
                      

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()