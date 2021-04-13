from threading import Thread
from concurrent import futures

import h5py
import time


import grpc
import Test_pb2 
import Test_pb2_grpc 
import cv2

import subprocess
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
import ImageProcess
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os

from embeddings import  Create_embeddings

from test import  Tpu_FaceRecognize
from config import*


Path = '/root/Test/copy/'



class BidirectionalService(Test_pb2_grpc.ImageSrvServicer):

    def analyze(self, request_iterator, context):
        
        print("Calling Function Analyzeing")
        
        for req in request_iterator:

                        
            img= req.data
            
           # print(type(img))

            file_name = req.name

            time_start = req.elapsed_time

            file_name = Path+file_name


            print("Image Location:"+file_name)

            nparr = np.fromstring(img, np.uint8)

            picture = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            cv2.imwrite(file_name,picture)

            ImageProcess.prediction(file_name,load_time,engine,labels,face_engine,class_arr,emb_arr)
            
        response = Test_pb2.Result(result="Sucess",elapsed_time=10,process_time=20)        
        
        return response
          


def loadmodel():

    global load_time
    global engine 
    global labels 
    global face_engine
    global class_arr
    global emb_arr
    
    
    load_time = time.time()
    print("load_time",type(load_time),load_time)    

  # Initialize engine.
    engine = DetectionEngine(Model_weight)    
    print("engine",type(engine),engine)
    labels = None
    print("labels",type(labels),labels)

  # Face recognize engine
    face_engine  = ClassificationEngine(FaceNet_weight)
    
    #print(face_engine)
    print("face_engine",type(face_engine),face_engine)
  # read embedding
    class_arr, emb_arr = ImageProcess.read_embedding(Embedding_book)
    print("class_arr",type(class_arr),class_arr)
    print("emb_arr",type(emb_arr),emb_arr)
    l =  time.time() - load_time
    
    return l


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    Test_pb2_grpc.add_ImageSrvServicer_to_server(BidirectionalService(), server)
    server.add_insecure_port('[::]:8080')

    print("------------------start Python Image Processing server")

    LoadingTime=loadmodel()

    print("--------------Model Loading Sucessful--------------")
    
    print('Load_model: {:.2f} sec'.format(LoadingTime))     

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
