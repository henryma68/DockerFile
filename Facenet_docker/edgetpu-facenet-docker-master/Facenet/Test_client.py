from __future__ import print_function

import grpc
import Test_pb2_grpc 
import Test_pb2 

#import Facenet

import time

import numpy as np
import cv2
import os 

Path = '/home/ubuntu/image/'
allFileList = os.listdir(Path)


def generaterequest(stub):

     #channel = grpc.insecure_channel('0.0.0.0:8080')
     #stub = Test_pb2_grpc.ImageSrvStub(channel)

     print("--------------Call Image Analyzing Begin--------------")

     def stream():

        for file in allFileList:             

            img=cv2.imread(Path+file)

            

            img_str = cv2.imencode('.jpg', img)[1].tostring()
            
            print('Start Transmition')

            request= Test_pb2.Image(data=img_str,name=file,elapsed_time=0)

            yield request
            

     response = stub.analyze(stream())

     print(response.result)   



def run():  
    with grpc.insecure_channel('127.0.0.1:8080') as channel:
        stub = Test_pb2_grpc.ImageSrvStub(channel)       

        generaterequest(stub)                  



if __name__ == '__main__':    
    run()

