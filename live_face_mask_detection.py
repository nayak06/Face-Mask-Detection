
from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab, Image
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from keras.preprocessing import image



model = tf.keras.models.load_model("model.h5",custom_objects={'GlorotUniform': glorot_uniform()})

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
source=cv2.VideoCapture(0)



labels_dict={0:'with_mask',1:'without_mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

'''
# -----------testing with single image----------
my_img = cv2.imread( 'me.jpg' , 0 )
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
faces=face_clsfr.detectMultiScale(my_img  , 1.3 , 5 )
i = 0 
print ( len( faces ) )
for x , y , w , h  in faces :
    face_img = my_img [ y : y + w , x : x + h ]
            #resized = cv2.resize (face_img , ( 100 , 100 ) )
    cv2.imwrite( str( i ) + '.jpg' , face_img )
    i += 1 







test_image = image.load_img('IMG_20190120_111555_HDR.jpg', target_size = (150 , 150 ) )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
label=np.argmax(result,axis=1)[0]
print( float (result) ) 
print(float( label))
#training_set.class_indices
'''
'''
'''
while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #my_img = cv2.imread( 'me.jpg' , 0 )
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    faces=face_clsfr.detectMultiScale(gray  , 1.3 , 5 )
    i = 0
    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        cv2.imwrite( str( i ) + '.jpg' , img )
        datapath = str( i ) + '.jpg'
        test_image = image.load_img( datapath , target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result=model.predict(test_image)
        label=np.argmax(result,axis=1)[0]
        print( float( result[ 0 ][ 0 ] ))
        #cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[result[ 0 ][ 0 ]],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[result[ 0 ][ 0 ]],-1)
        cv2.putText(img, labels_dict[result[ 0 ][ 0 ]], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
