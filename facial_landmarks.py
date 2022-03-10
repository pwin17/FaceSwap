#!/usr/bin/python3
import cv2
import numpy as np
import dlib
import random 

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True


def get_faces(img):
    altFaceDetector = dlib.get_frontal_face_detector()
    # dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rects = dnnFaceDetector(img_gray, 1)
    rects = altFaceDetector(img_gray, 1)
    # rects1 = dnnFaceDetector(img_gray, 1)
    # print(rects[0])
    # print(rects[0].bottom())

    return rects

def draw_rects(rects,img_orig):
    img = img_orig.copy()
    for r in rects:
        two_pts = [(r.rect.left(),r.rect.top()), (r.rect.right(),r.rect.bottom())]
        cv2.rectangle(img,two_pts[0],two_pts[1],(0,255,0),3)
    return img

def draw_rects_frontal(rects,img_orig):
    img = img_orig.copy()
    for r in rects:
        two_pts = [(r.left(),r.top()),(r.right(),r.bottom())]
        cv2.rectangle(img,two_pts[0],two_pts[1],(0,255,0),3)
    return img

def get_facial_landmarks(img,rects):
    landmark_detector = dlib.shape_predictor("./packages/shape_predictor_68_face_landmarks.dat")
    final_shapes = []
    for rect in rects:
        shape = landmark_detector(img,rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        final_shapes.append(shape_np)
    return final_shapes

def draw_facial_landmarks(img_orig,final_shapes):
    img = img_orig.copy()
    # final_shapes = [final_shapes[1]]
    for shape in final_shapes:
        for coord in shape:
            cv2.circle(img,coord,2,(0,0,255,-1))
    return img

def draw_delaunay(img_orig, subdiv, delaunay_color ) :

    img = img_orig.copy()
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    print("triangle0: ",triangleList[0])
    for t in triangleList :

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    return triangleList,img

def get_delaunay_triangulation(img,final_shapes):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv  = cv2.Subdiv2D(rect)
    pt_list = []
    j=0

    for i in range(len(final_shapes[0])):
        pt = (int(final_shapes[0][i][0]),int(final_shapes[0][i][1]))
        subdiv.insert(pt)

    return subdiv

def getBarycentricMatrix(trianglePts):
    B = np.array([[trianglePts[0],trianglePts[2],trianglePts[4]],
                  [trianglePts[1],trianglePts[3],trianglePts[5]],
                  [1,1,1]])
    return B


# Of a given triangle and its B matrix, it returns the points inside the triange and the barycentric coordinates
def calculateBarycentricCoords(img, B, trianglepts):

    pt1 = (int(trianglepts[0]), int(trianglepts[1]))
    pt2 = (int(trianglepts[2]), int(trianglepts[3]))
    pt3 = (int(trianglepts[4]), int(trianglepts[5]))
    x,y,w,h = cv2.boundingRect(np.array([pt1,pt2,pt3]))
    print("bounding rect:",x,y,x+w,y+h)
    
    xx,yy = np.indices((w, h))
    xx = xx + x
    yy = yy + y
    # print("xx len: ",xx.shape)
    # print("yy len: ",yy.shape)
    B_inv = np.linalg.inv(B)
    bary_matrix = []

    x_valid = []
    y_valid = []
    bary_valid = []
    for x,y in zip(xx.ravel(),yy.ravel()):
        img_coord = np.reshape([x,y,1],(3,1))
        # print("img coord: ",img_coord)
        bary_coord = np.matmul(B_inv,img_coord)
        # print("bary coord: ",bary_coord)
        bary_matrix.append(bary_coord)

        # Filter to keep valid barycentric coordinates

        if(bary_coord[0]>0 and bary_coord[0]<=1 and bary_coord[1]>0 and bary_coord[1]<=1 and \
            bary_coord[2]>0 and bary_coord[2]<=1 and 0.999999<=bary_coord[0]+bary_coord[1]+bary_coord[2]<=1.000001):
            x_valid.append(x)
            y_valid.append(y)
            bary_valid.append(bary_coord)
            # cv2.circle(img,(x,y),1,(255,0,0),-1)
    # cv2.imshow('img,barycentric',img)
    # cv2.waitKey()
    return x_valid, y_valid, bary_valid 


def getSourceLocations(A, bary_coords,img_src):
    x_source = []
    y_source = []
    for bary_coord in bary_coords:
        cart_coord = np.matmul(A,bary_coord)
        x = int(cart_coord[0]/cart_coord[2])
        y = int(cart_coord[1]/cart_coord[2])
        x_source.append(x)
        y_source.append(y)
        # cv2.circle(img_src,(x,y),1,(0,0,255),1)
    # cv2.imshow('extraction points',img_src)
    # cv2.waitKey(0)
    return x_source,y_source

def copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst):
    # cv2.imshow("before copying src:",img_src)
    # cv2.imshow("before copying dst:",img_dst)
    print("x_source:",len(x_source))
    for i in range(len(x_source)):
        img_dst[y_target[i]][x_target[i]] = img_src[y_source[i]][x_source[i]]
    # cv2.imshow("after copying src:",img_src)
    # cv2.imshow("after copying dst:",img_dst)
    return img_dst
    # cv2.waitKey()

def check_triangulation_order(img_src,triangleList_src,img_dst,triangleList_dst):
    
    for t_s,t_d in zip(triangleList_src,triangleList_dst) :
        img_src_copy = img_src.copy()
        pt1 = (int(t_s[0]), int(t_s[1]))
        pt2 = (int(t_s[2]), int(t_s[3]))
        pt3 = (int(t_s[4]), int(t_s[5]))

        # if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        cv2.line(img_src_copy, pt1, pt2, (255,0,0), 1, cv2.LINE_AA, 0)
        cv2.line(img_src_copy, pt2, pt3, (255,0,0), 1, cv2.LINE_AA, 0)
        cv2.line(img_src_copy, pt3, pt1, (255,0,0), 1, cv2.LINE_AA, 0)

        img_dst_copy = img_dst.copy()
        pt1d = (int(t_d[0]), int(t_d[1]))
        pt2d= (int(t_d[2]), int(t_d[3]))
        pt3d = (int(t_d[4]), int(t_d[5]))

        # if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        cv2.line(img_dst_copy, pt1d, pt2d, (0,0,255), 1, cv2.LINE_AA, 0)
        cv2.line(img_dst_copy, pt2d, pt3d, (0,0,255), 1, cv2.LINE_AA, 0)
        cv2.line(img_dst_copy, pt3d, pt1d, (0,0,255), 1, cv2.LINE_AA, 0)
        cv2.imshow('src triangles',img_src_copy)
        cv2.imshow('dst triangles',img_dst_copy)

        cv2.waitKey(0)
        # cv2.destroyWindow('')

def main():
    img_src = cv2.imread('./data/ron.jpg')
    print(img_src.shape)
    img_src =cv2.resize(img_src,(500,500))
    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_src)
    final_shapes = get_facial_landmarks(img_src_gray,rects)
    # print((final_shapes[0]).shape)
    
    landmark_img = draw_facial_landmarks(img_src,final_shapes)
    
    subdiv = get_delaunay_triangulation(img_src,final_shapes)

    triangleList_src, delaunay_img = draw_delaunay(img_src, subdiv, (255,255,255))
    # print("len",len(triangleList_src[0]))

    # cv2.imshow('face landmarks',landmark_img)
    cv2.imshow('delaunay_img',delaunay_img)

    # -------------------------------------------------------
    img_dst = cv2.imread('./data/mes.jpg')
    print(img_dst.shape)
    img_dst =cv2.resize(img_dst,(500,500))
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_dst)
    final_shapes = get_facial_landmarks(img_dst_gray,rects)
    # print((final_shapes[0]).shape)
    
    landmark_img = draw_facial_landmarks(img_dst,final_shapes)
    
    subdiv = get_delaunay_triangulation(img_dst,final_shapes)

    triangleList_dst, delaunay_img = draw_delaunay(img_dst, subdiv, (255,255,255))

    # cv2.imshow('face landmarks_src',landmark_img)
    cv2.imshow('delaunay_img_dst',delaunay_img)
    cv2.waitKey()

    check_triangulation_order(img_src,triangleList_src,img_dst,triangleList_dst)

    cv2.imshow('img_dst before warping:',img_dst)


    # Execute this process for every triangle:
    for i in range(len(triangleList_dst)):
        B = getBarycentricMatrix(triangleList_dst[i])
        x_target, y_target, bary_valid= calculateBarycentricCoords(img_dst,B,triangleList_dst[i])
        # print("valid points: ")
        # for i in range(len(x_valid)):
        #     print(x_valid[i],y_valid[i])
        # print("len bary valid:",len(bary_valid))

        A = getBarycentricMatrix(triangleList_src[i])
        x_source,y_source = getSourceLocations(A, bary_valid,img_src)
        img_dst = copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst)
    # print("Source X:",len(x_source))
    cv2.imshow('img_dst after warping:',img_dst)
    cv2.waitKey(0)
    # print(a)
if __name__=="__main__":
    main()