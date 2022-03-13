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
            cv2.circle(img,coord,2,(0,0,255),-1)
            count += 1
    return img

def get_faces(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # altFaceDetector = dlib.get_frontal_face_detector()
    # rects = altFaceDetector(img_gray, 1)

    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    rects = dnnFaceDetector(img_gray, 1)

    
   
    

    return rects

def main():

    method = "TPS" #"TRI"
    img_src = cv2.imread('./data/impractical_jokers.jpg')
    img_dst = cv2.imread('./data/impractical_jokers.jpg')
    img_src_original = img_src.copy()
    # cv2.imshow('img_src',img_src)

    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_src)

    img_dst_original = img_dst.copy()
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_dst)
    final_shapes_dst = get_facial_landmarks(img_dst_gray,[rects[1]])
    landmark_img_dst = draw_facial_landmarks(img_dst,final_shapes_dst)
    final_shapes_src = get_facial_landmarks(img_src_gray,[rects[0]])
    landmark_img_src = draw_facial_landmarks(img_src,final_shapes_src)
    src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)
    if method=="TRI":
        img_swap1 = TRI(img_src,img_dst,final_shapes_src,final_shapes_dst)
        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_src, dst_mask,center_dst, cv2.NORMAL_CLONE)
        img_swap2 = TRI(img_src,img_swap1,final_shapes_dst,final_shapes_src)
        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, src_mask, center_src, cv2.NORMAL_CLONE)
        cv2.imshow('blended final_img3',img_swap2)

    elif method == "TPS":
        
        img_swap1 = TPS(img_src,img_dst,final_shapes_src,final_shapes_dst,img_src_original)
        cv2.imshow('one face swap,b4 blend',img_swap1)
        cv2.imshow('src img',img_src)
        cv2.imshow('dst img',img_dst)
        
        cv2.imshow('src mask',src_mask)
        cv2.imshow('dst mask',dst_mask)


        img_swap1 = cv2.seamlessClone(np.uint8(img_swap1),img_src, src_mask,center_src, cv2.NORMAL_CLONE)

        cv2.imshow('blended final_img1',img_swap1)
        img_swap2 = TPS(img_swap1,img_src,final_shapes_dst,final_shapes_src,img_swap1)
        cv2.imshow('two face swap,b4 blend',img_swap2)

        img_swap2 = cv2.seamlessClone(np.uint8(img_swap2),img_swap1, dst_mask, center_dst, cv2.NORMAL_CLONE)

        cv2.imshow('blended final_img2',img_swap2)
        # cv2.imshow('img src tps2',img_src)
        # cv2.imshow('img_swap2',img_swap2)
        # cv2.imshow('img_swap1',img_swap1)

        # cv2.waitKey(0)

    # cv2.imshow('warp2',img_swap2)
    # src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)
    # cv2.imshow('src mask',src_mask)
    # combined_mask = cv2.bitwise_not(cv2.bitwise_or(src_mask,dst_mask))
    # cv2.imshow('combined mask',combined_mask)
    # no_face_img = cv2.bitwise_and(combined_mask,img_src)
    # cv2.imshow('no faces',no_face_img)
    
    # final_img2 = cv2.seamlessClone(np.uint8(final_img1), np.uint8(final_img1), dst_mask, center_dst, cv2.NORMAL_CLONE)
    # cv2.imshow('final output2',final_img2)
    cv2.waitKey(0)


if __name__=="__main__": 
    main()