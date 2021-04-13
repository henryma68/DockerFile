import cv2
import h5py
import time
import platform
import numpy
import subprocess
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from embeddings import  Create_embeddings
import numpy as np
from test import  Tpu_FaceRecognize
from config import*


def crop_image(ans, frame):
    Images_cropped = []
    for i in range(0, len(ans)):
        img_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        BBC = ans[i].bounding_box                   # bounding_box_coordinate

        x = int(BBC[0][0])
        y = int(BBC[0][1])
        w = int(BBC[1][0] - BBC[0][0])
        h = int(BBC[1][1] - BBC[0][1])

        img_crop = img_crop[y:y+h, x:x+w]

        img_crop = cv2.resize(img_crop, (160, 160))

        Images_cropped.append(img_crop)

    return Images_cropped


def read_embedding(path=Embedding_book):
    
    try:
        f=h5py.File(path,'r')
    except OSError:
        face_engine  = ClassificationEngine(FaceNet_weight)
        Create_embeddings(face_engine)
        f=h5py.File(path, 'r')

    class_arr=f['class_name'][:]
    class_arr=[k.decode() for k in class_arr]
    emb_arr=f['embeddings'][:]

    return class_arr, emb_arr



def prediction(file_name,load_time,engine,labels,face_engine,class_arr,emb_arr):
  
  file_name = file_name
  load_time = load_time
  engine = engine
  labels = labels
  face_engine = face_engine
  class_arr = class_arr
  emb_arr = emb_arr

  with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:     

            frame = cv2.imread(file_name)
            print('file_name:', file_name)                 
            img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)   

            # cap = cv2.VideoCapture(0)

            t1 = cv2.getTickCount()
            ans = engine.detect_with_image(
                    img,
                    threshold=0.05,
                    keep_aspect_ratio=False,
                    relative_coord=False,
                    top_k=10)
            img = numpy.asarray(img) 

            print('img:', img)

            if ans:
    
                crop_img = crop_image(ans, frame)
                embs = Tpu_FaceRecognize(face_engine, crop_img)
                face_num = len(ans)
                face_class = ['Others']*face_num
                for i in range(face_num):
                    diff = np.mean(np.square(embs[i]-emb_arr), axis=1)
                    min_diff = min(diff)
                    if min_diff < THRED:
                        index = np.argmin(diff)

                        face_class[i] = class_arr[index]                 
                      
                print('Face_class:', face_class)                         
                print('Classes:', class_arr) 
                """

                # If the input picture is not categorized, let user input the name
                if 'Others' in face_class:
                    # print("usagi")
                    for k in range(0, len(crop_img)):
                        new_class_name = input('Please input your name of class:')
                        new_save = cv2.cvtColor(crop_img[k], cv2.COLOR_BGR2RGB)
                        cv2.imwrite('pictures/' + str(new_class_name) + '.jpg', new_save)

                    Create_embeddings(face_engine)
                    class_arr, emb_arr = read_embedding('embedding_book/embeddings.h5')  
                    
                    """

                
                for count, obj in enumerate(ans):
                          
                    print('-----------------------------------------')
                    if labels:
                        print(labels[obj.label_id])
                          
                    print('Score = ', obj.score)
                    box = obj.bounding_box.flatten().tolist()
                    # Draw a rectangle and label
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                    cv2.putText(img, '{}'.format(face_class[count]), (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 0, 0), 1, cv2.LINE_AA)

                    
            t2 = cv2.getTickCount()
            t = (t2-t1)*1000/cv2.getTickFrequency()
            cv2.putText(img, 'inf_time: {:.2f}'.format(t), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(img, 'A: Add new class', (5, 450), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, 'Q: Quit', (5, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            img_ = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 
            # cv2.imshow('frame', img_)
            cv2.imwrite(file_name, img_)        

            




                         
















