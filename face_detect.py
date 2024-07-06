import cv2

import os

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('H:\PROJ\database connection\haarcascade_frontalface_default.xml')

count=0

nameID=str(input("Enter your Name:")).lower()

path='image/'+ nameID

#checks if it already exists or not

isExist=os.path.exists(path)

if isExist:
    print("name already taken")
    nameID=str(input("enter your name again"))
else:
    os.makedirs(path)

#recording begins here

while True:
    ret,frame=video.read()
    faces=facedetect.detectMultiScale(frame,1.3, 5)
    for x,y,w,h in faces:
        count=count+1
        name= os.path.join(path, f"{count}.jpg")
        print("creating Images..........."+name)
        cv2.imwrite(name,frame[y:y+h,x:x+w])
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)  
    cv2.imshow('WindowFrame',frame)
    cv2.waitKey(1)
    
    
    if count>=50:
        break
video.release()
cv2.destroyAllWindows()
