#!/usr/bin/python3
import cv2
import numpy as np
import dlib
import random 
from tri import TRI
from TPS import TPS
from utils import hull_masks

def get_facial_landmarks(img,rects):
    landmark_detector = dlib.shape_predictor("./packages/shape_predictor_68_face_landmarks.dat")
    final_shapes = []
    for rect in rects:
        shape = landmark_detector(img,rect.rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        final_shapes.append(shape_np)
    return final_shapes

def draw_facial_landmarks(img_orig,final_shapes):
    img = img_orig.copy()

    count = 1
    for shape in final_shapes:
        for coord in shape:
            cv2.circle(img,tuple(coord),2,(0,0,255),-1)
            count += 1
    return img

def get_faces(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # altFaceDetector = dlib.get_frontal_face_detector()
    # rects = altFaceDetector(img_gray, 1)

    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    rects = dnnFaceDetector(img_gray, 1)
    return rects

def SwapFaces(img_src,img_dst,method,swap_logic):

    # method = "TPS" #"TRI"
    # img_src = cv2.imread('./data/put-zel.jpg')
    # img_dst = cv2.imread('./data/put-zel.jpg')
    # cv2.imshow('src',img_src)
    # cv2.imshow('dst',img_dst)
    # cv2.waitKey(0)

    # img_src = cv2.resize(img_src,(img_src.shape[1]//2,img_src.shape[0]//2))
    # img_dst = cv2.resize(img_dst,(img_dst.shape[1]//2,img_dst.shape[0]//2))

    # img_dst = img_dst.resize((img_dst.shape[0]//2,img_dst.shape[1]//2))

    # img_src = frame.copy()
    # img_dst = frame.copy()

    img_src_original = img_src.copy()

    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    img_dst_original = img_dst.copy()
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    src_rects= get_faces(img_src)
    dst_rects= get_faces(img_dst)
    num_faces = len(src_rects)
    
    #Scenario 1 : two ima
    if swap_logic == "swap_within_frame":
        src_idx = 0
        dst_idx = 1
    else:
        src_idx = 0
        dst_idx = 0

    final_shapes_dst = get_facial_landmarks(img_dst_gray,[dst_rects[dst_idx]])
    landmark_img_dst = draw_facial_landmarks(img_dst,final_shapes_dst)

    final_shapes_src = get_facial_landmarks(img_src_gray,[src_rects[src_idx]])
    landmark_img_src = draw_facial_landmarks(img_src,final_shapes_src)

    src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)
    
    if method=="TRI":
        img_swap1 = TRI(img_src,img_dst,final_shapes_src,final_shapes_dst)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_src, dst_mask,center_dst, cv2.NORMAL_CLONE)
        
        img_swap2 = TRI(img_src,img_swap1,final_shapes_dst,final_shapes_src)
        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, src_mask, center_src, cv2.NORMAL_CLONE)
        cv2.imshow('blended final_img2',img_swap2)

    elif method == "TPS":
        
        img_swap1 = TPS(img_src,img_dst,final_shapes_src,final_shapes_dst,img_src_original)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_src, src_mask,center_src, cv2.NORMAL_CLONE)

        img_swap2 = TPS(img_swap1,img_src,final_shapes_dst,final_shapes_src,img_swap1)
        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, dst_mask, center_dst, cv2.NORMAL_CLONE)
        img_swap2 = cv2.resize(img_swap2,(img_swap2.shape[1]*2,img_swap2.shape[0]*2),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('blended final_img2',img_swap2)

    cv2.waitKey(0)

# 2 imgs, 1 img and 1 vid, 1 vid and None
def parse_input_types(input1,input2):
    video_formats = ['mp4','avi','mov']
    image_formats = ['jpg','jpeg','png']
    input1_format = None
    input2_format = None
    if input1 is not None:
        input1_format = input1.split('.')[-1]
    if input2 is not None:
        input2_format = input2.split('.')[-1]

    input_formats = [input1_format,input2_format]
    input_types = [None,None]
    for i in range(len(input_formats)):
        if(input_formats[i] in image_formats):
            input_types[i] = 'img'
        elif input_formats[i] in video_formats:
            input_types[i] = 'vid'

    if input_types[0]=='img' and input_types[1]=='img':
        swap_logic ="swp_two_imgs"
    elif (input_types[0]=='vid' and input_types[1]=='img') :
        swap_logic = "swap_img_in_vid"
    elif (input_types[0]=='vid' and input_types[1] is None):
        swap_logic = "swap_within_frame"
    else:
        swap_logic = None

    return swap_logic
if __name__=="__main__": 

    # input1 = './data/twofaces.mp4'
    # input2 = None

    input1 = './data/mes.jpg'
    input2 = './data/ron.jpg'
    method = "TPS"


    swap_logic = parse_input_types(input1,input2)
    if swap_logic is None:
        print("Invalid input formats. Please ensure input2 is an image if input 1 is an img, and an image or None if input1 is a video")
        exit()    

    elif(swap_logic=="swap_within_frame" or swap_logic=="swap_img_in_vid"):
        cap = cv2.VideoCapture(input1)
        img_dst = cv2.imread(input2)
        while(True):
            ret,img_src = cap.read()

            if ret:
                if swap_logic == "swap_within_frame":
                    img_dst = img_src.copy()
                SwapFaces(img_src,img_dst,method,swap_logic)

            else:
                print("Video completed")
                break
    else:
        img_src = cv2.imread(input1)
        img_dst = cv2.imread(input2)
        # img_src = cv2.resize(img_src,(500,500))
        # img_dst = cv2.resize(img_dst,(500,500))

        SwapFaces(img_src,img_dst,method,swap_logic)