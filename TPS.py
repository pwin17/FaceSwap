#!/usr/bin/python3
import cv2
import numpy as np
from utils import *

def get_TPS_params(src_shapes,dst_shapes):
    # 68 x 2 shape
    U = lambda r: (r**2)*np.log(r**2)

    K = np.zeros((len(src_shapes),len(src_shapes)))
    l = 0.00000001
    # print("src shapes like:",(src_shapes).shape)
    for i in range(len(src_shapes)):
        for j in range(len(src_shapes)):
            K[i,j] = np.linalg.norm(src_shapes[i][:] - src_shapes[j][:])
            K[i,j] = U(K[i,j]+l)
    P = np.hstack((src_shapes, np.ones((len(src_shapes), 1))))
    I = np.identity(len(src_shapes)+3)

    col1 = np.vstack((K,P.T))
    col2 = np.vstack((P, np.zeros((3,3))))

    M = np.hstack((col1,col2))
    lI = l*I
    # print("lI shape:", np.shape(lI))
    M = M+lI
    Minv = np.linalg.inv(M)
    # print("M inv: ", np.shape(Minv))
    # # [x1, x2, x3,...., xp, 0,0,0]
    dst_x = np.hstack((dst_shapes[:,0],[0,0,0])).T
    params_x = np.matmul(Minv,dst_x)
    # print("params_x shape:", np.shape(params_x))
    dst_y = np.hstack((dst_shapes[:,1],[0,0,0])).T
    params_y = np.matmul(Minv,dst_y)
    # print("params_y shape:", np.shape(params_y))
    return params_x, params_y,M

def warp_TPS(src_shapes,dst_shapes,img_src,img_dst,src_hull,dst_hull,params_x,params_y,M):
    interior_points = []
    img_src_hull = img_src.copy()
    # cv2.drawContours(img_src_hull,[src_hull],-1,(0,0,255),4)
    # cv2.imshow("hull viz",img_src_hull)
    # cv2.waitKey(0)
    # cv2.imshow("warp_TPS src:",img_src_hull)
    # cv2.imshow("warp_TPS dst:",img_dst)
    # cv2.waitKey()
    U = lambda r: (r**2)*np.log(r**2)
    for i in range(img_src.shape[0]):
        for j in range(img_src.shape[1]):
            # print("i,j: ",i,j)
            if(cv2.pointPolygonTest(src_hull,(j,i),False) >=0):
                # print("added interior pts")
                img_src_hull[i,j]=[0,255,0]
                interior_points.append([j,i])
    cv2.imshow('hull color',img_src_hull)
    cv2.waitKey()
    interior_points = np.array(interior_points)
    
    K = np.zeros((interior_points.shape[0],src_shapes.shape[0]))
    for i in range(interior_points.shape[0]):
        for j in range(src_shapes.shape[0]):
            K[i,j] = np.linalg.norm(interior_points[i,:] - src_shapes[j,:]) + 0.00000001
            K[i,j] = U(K[i,j])
    print("interior pts:",interior_points.shape)
    P = np.hstack((interior_points, np.ones((len(interior_points), 1))))
    I = np.identity(len(interior_points)+3)
    l = 0.00000001
  
    params = np.vstack((params_x[:-3],params_y[:-3]))
    loc_sum1 = np.matmul(K,params.T)
    P_params = np.vstack((params_x[-3:],params_y[-3:]))
    loc_sum2 = np.matmul(P,P_params.T)
    locs = loc_sum1 + loc_sum2
   
    return locs, interior_points

def TPS(img_src,img_dst,final_shapes_src,final_shapes_dst,img_src_original):
    
    final_shapes_src = np.reshape(final_shapes_src, (68,2)).astype('int32')
    final_shapes_dst = np.reshape(final_shapes_dst, (68,2)).astype('int32')
    # print(np.shape(final_shapes_src))
    # print(np.shape(final_shapes_dst))

    src_hull,src_mask,center_src,dst_hull,dst_mask,center_dst = hull_masks(img_src,img_dst,final_shapes_src,final_shapes_dst)

    params_x, params_y, M = get_TPS_params(final_shapes_src, final_shapes_dst)
    # cv2.imshow('src hull',src_hull)
    # print("src hull:", src_hull)
    
    locs,interior_pts  = warp_TPS(final_shapes_src,final_shapes_dst,img_src,img_dst,src_hull,dst_hull,params_x,params_y,M)
    locs = locs.astype(np.int32)
    img_src_copy = img_src.copy()
    interior_pts = interior_pts.astype(np.int32)
    warp_TPS_img = copyPixelsTPS(x_source=locs[:,1],y_source=locs[:,0],x_target=interior_pts[:,1],y_target=interior_pts[:,0],\
        img_src=img_dst,img_dst=img_src_copy)

    
    # cv2.imshow('blended_img:',final_img)

    # cv2.waitKey(0)
    return warp_TPS_img
