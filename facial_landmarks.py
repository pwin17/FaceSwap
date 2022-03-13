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
    # altFaceDetector = dlib.get_frontal_face_detector()
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    print('loaded detector')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img dnn',img_gray)
    cv2.waitKey(0)
    rects = dnnFaceDetector(img_gray, 1)
    print('rects loaded')
    # rects = altFaceDetector(img_gray, 1)
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
        shape = landmark_detector(img,rect.rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        final_shapes.append(shape_np)
    return final_shapes

def draw_facial_landmarks(img_orig,final_shapes):
    img = img_orig.copy()

    # final_shapes = [final_shapes[1]]
    count = 1
    for shape in final_shapes:
        for coord in shape:
            cv2.circle(img,coord,2,(0,0,255,-1))
            # cv2.putText(img,str(count),coord, cv2.FONT_HERSHEY_SIMPLEX, 0.25,  (0,0,255), 1, cv2.LINE_AA)
            count += 1

    return img

def draw_delaunay(img_orig, subdiv, delaunay_color ) :

    img = img_orig.copy()
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    # print("triangle0: ",triangleList[0])
    count = 0
    # print(np.shape(triangleList), "triangle list")
    for t in triangleList :

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            count += 1
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    
    # print("valid triangles: ",count)
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
    # print("bounding rect:",x,y,x+w,y+h)
    
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
    # print("x_source:",len(x_source))
    for i in range(len(x_source)):
        # y_target[i] = min(max(y_target[i],0),img_dst.shape[0]-1)
        # x_target[i] = min(max(x_target[i],0),img_dst.shape[0]-1)

        img_dst[y_target[i]][x_target[i]] = img_src[y_source[i]][x_source[i]]
    # cv2.imshow("after copying src:",img_src)
    # cv2.imshow("after copying dst:",img_dst)
    # cv2.imshow("after copying src:",img_src)
    
    # cv2.waitKey()
    return img_dst

def copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst):
    # cv2.imshow("before copying src:",img_src)
    # cv2.imshow("before copying dst:",img_dst)
    # print("x_source:",len(x_source))
    for i in range(len(x_source)):
        # y_target[i] = min(max(y_target[i],0),img_dst.shape[0]-1)
        # x_target[i] = min(max(x_target[i],0),img_dst.shape[0]-1)

        img_dst[y_target[i]][x_target[i]] = img_src[y_source[i]][x_source[i]]
    # cv2.imshow("after copying src:",img_src)
    # cv2.imshow("after copying dst:",img_dst)
    # cv2.imshow("after copying src:",img_src)
    
    # cv2.waitKey()
    return img_dst


def sort_triangles(triangleList_src, src_shapes, triangleList_dst, dst_shapes):
    dst_shapes = np.reshape(dst_shapes, (68,2))
    src_shapes = np.reshape(src_shapes, (68,2))
    new_triangles_src = []
    # print("dst_shapes[0]", dst_shapes[0])
    for d in triangleList_dst:
        pt1 = [d[0], d[1]]
        pt2 = [d[2], d[3]]
        pt3 = [d[4], d[5]]
        pt1_s, pt2_s, pt3_s = [], [], []
        
        for i in range(len(dst_shapes)):
            
            if dst_shapes[i][0] == d[0] and dst_shapes[i][1] == d[1]:
                pt1_s = src_shapes[i]
            elif  dst_shapes[i][0] == d[2] and dst_shapes[i][1] == d[3]:
                pt2_s = src_shapes[i]
            elif  dst_shapes[i][0] == d[4] and dst_shapes[i][1] == d[5]:
                pt3_s = src_shapes[i]
        if len(pt1_s) ==0 or len(pt2_s) == 0 or len(pt3_s) == 0:
            pass
        else:
            new = [pt1_s[0], pt1_s[1], pt2_s[0], pt2_s[1], pt3_s[0], pt3_s[1]]
            new_triangles_src.append(new)
    return new_triangles_src

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

# def get_P()
def get_TPS_params(src_shapes,dst_shapes):
    # 68 x 2 shape
    U = lambda r: (r**2)*np.log(r**2)
    K = np.zeros((len(src_shapes),len(src_shapes)))
    l = 0.00000001
    print("src shapes like:",(src_shapes).shape)
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
    print("M inv: ", np.shape(Minv))
    # # [x1, x2, x3,...., xp, 0,0,0]
    dst_x = np.hstack((dst_shapes[:,0],[0,0,0])).T
    params_x = np.matmul(Minv,dst_x)
    print("params_x shape:", np.shape(params_x))
    dst_y = np.hstack((dst_shapes[:,1],[0,0,0])).T
    params_y = np.matmul(Minv,dst_y)
    print("params_y shape:", np.shape(params_y))
    return params_x, params_y,M


def warp_TPS(src_shapes,dst_shapes,img_src,img_dst,src_hull,dst_hull,params_x,params_y,M):
    interior_points = []
    U = lambda r: (r**2)*np.log(r**2)
    for i in range(img_src.shape[0]):
        for j in range(img_src.shape[1]):
            if(cv2.pointPolygonTest(src_hull,(i,j),False) >=0):
                interior_points.append([i,j])
    interior_points = np.array(interior_points)
    print("interior pts:",interior_points.shape)
    
    K = np.zeros((interior_points.shape[0],src_shapes.shape[0]))
    for i in range(interior_points.shape[0]):
        for j in range(src_shapes.shape[0]):
            K[i,j] = np.linalg.norm(interior_points[i,:] - src_shapes[j,:]) + 0.00000001
            K[i,j] = U(K[i,j])

    print("K shape:",K.shape)
    P = np.hstack((interior_points, np.ones((len(interior_points), 1))))
    I = np.identity(len(interior_points)+3)
    l = 0.00000001
    print("P shape",P.shape)
    # col1 = np.vstack((K,P.T))
    # print("col1 shape:",col1.shape)
    # col2 = np.vstack((P, np.zeros((3,3))))
    # M = np.hstack((col1,col2))
    # lI = l*I
    # print("lI shape:", np.shape(lI))
    # M = M+lI
    params = np.vstack((params_x[:-3],params_y[:-3]))
    # ax_x,ay_x,a1_x = params_x[-3],params_x[-2],params_x[-1]
    # ax_y,ay_y,a1_y = params_y[-3],params_y[-2],params_y[-1]
    print("params shape:",params.shape)
    loc_sum1 = np.matmul(K,params.T)
    P_params = np.vstack((params_x[-3:],params_y[-3:]))
    loc_sum2 = np.matmul(P,P_params.T)

    locs = loc_sum1 + loc_sum2
    print("locs shape:",locs.shape)    
    return locs, interior_points
    # print("M shape:",M.shape)

def warp_TPS2(src_shapes,dst_shapes,img_src,img_dst,src_hull,dst_hull,params_x,params_y,M):
    interior_points = []
    U = lambda r: (r**2)*np.log(r**2)
    for i in range(img_dst.shape[0]):
        for j in range(img_dst.shape[1]):
            if(cv2.pointPolygonTest(dst_hull,(i,j),False) >=0):
                interior_points.append([i,j])
    interior_points = np.array(interior_points)
    print("interior pts:",interior_points.shape)
    
    K = np.zeros((interior_points.shape[0],dst_shapes.shape[0]))
    for i in range(interior_points.shape[0]):
        for j in range(dst_shapes.shape[0]):
            K[i,j] = np.linalg.norm(interior_points[i,:] - src_shapes[j,:]) + 0.000001
            K[i,j] = U(K[i,j])

    print("K shape:",K.shape)
    P = np.hstack((interior_points, np.ones((len(interior_points), 1))))
    print("P shape",P.shape)
    params = np.vstack((params_x[:-3],params_y[:-3]))
    # ax_x,ay_x,a1_x = params_x[-3],params_x[-2],params_x[-1]
    # ax_y,ay_y,a1_y = params_y[-3],params_y[-2],params_y[-1]
    print("params shape:",params.shape)
    loc_sum1 = np.matmul(K,params.T)
    P_params = np.vstack((params_x[-3:],params_y[-3:]))
    loc_sum2 = np.matmul(P,P_params.T)

    locs = loc_sum1 + loc_sum2
    print("locs shape:",locs.shape)    
    return locs, interior_points

def main():

    method = "TRI" #"TRI"


    img_src = cv2.imread('./data/two_people.jpg')
    img_src_original = img_src.copy()
    cv2.imshow('img_src',img_src)
    cv2.waitKey(0)
    # print(img_src.shape)
    # img_src =cv2.resize(img_src,(500,500))
    # img_src_original = img_src.copy()
    img_src_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_src)
    print('rects obtained')
    final_shapes_src = get_facial_landmarks(img_src_gray,[rects[0]])
    # print((final_shapes[0]).shape)
    
    landmark_img_src = draw_facial_landmarks(img_src,final_shapes_src)
    # print('src landmarks: ',np.shape(final_shapes_src))
    subdiv = get_delaunay_triangulation(img_src,final_shapes_src)

    triangleList_src, delaunay_img = draw_delaunay(img_src, subdiv, (255,255,255))
    

    cv2.imshow('src face landmarks',landmark_img_src)
    # if cv2.waitKey(0)==ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.imshow('delaunay_img',delaunay_img)

    # -------------------------------------------------------

    img_dst = cv2.imread('./data/two_people.jpg')

    # print(img_dst.shape)
    # img_dst =cv2.resize(img_dst,(500,500))
    img_dst_original = img_dst.copy()
    img_dst_gray = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)

    rects= get_faces(img_dst)
    final_shapes_dst = get_facial_landmarks(img_dst_gray,[rects[1]])
    # print('dst landmarks: ',np.shape(final_shapes_dst))


    landmark_img_dst = draw_facial_landmarks(img_dst,final_shapes_dst)

    # cv2.imshow('dst face landmarks',landmark_img_dst)
    # if cv2.waitKey(0)==ord('q'):
    #     cv2.destroyAllWindows()

    if method=="TRI":
    
        subdiv = get_delaunay_triangulation(img_dst,final_shapes_dst)

        triangleList_dst, delaunay_img = draw_delaunay(img_dst, subdiv, (255,255,255))

        # triangleList_src = triangleList_src[1:]

        print("len",np.shape(triangleList_src))
        print("len",np.shape(triangleList_dst))

        triangleList_src = sort_triangles(triangleList_src, final_shapes_src, triangleList_dst, final_shapes_dst)

        print("len",np.shape(triangleList_src))
        print("len",np.shape(triangleList_dst))

        # cv2.imshow('face landmarks_src',landmark_img)
        cv2.imshow('delaunay_img_dst',delaunay_img)
        cv2.waitKey()

        # check_triangulation_order(landmark_img_src,triangleList_src,landmark_img_dst,triangleList_dst)

        cv2.imshow('img_dst before warping123:',img_dst)
        cv2.waitKey(0)

        # Execute this process for every triangle:
        for i in range(len(triangleList_dst)):
            B = getBarycentricMatrix(triangleList_dst[i])
            x_target, y_target, bary_valid= calculateBarycentricCoords(img_dst,B,triangleList_dst[i])
            # print("valid points: ")
            # for i in range(len(x_valid)):
            #     print(x_valid[i],y_valid[i])
            # print("len bary valid:",len(bary_valid))
        # print(final_shapes_src[:,0])
            A = getBarycentricMatrix(triangleList_src[i])
            x_source,y_source = getSourceLocations(A, bary_valid,img_src)
            img_dst = copyPixels(x_source,y_source,x_target,y_target,img_src,img_dst)
        # exit()
        cv2.imshow("final warped img",img_dst)
        cv2.waitKey(0)
        # print(a)
    elif method == "TPS":
        
        final_shapes_src = np.reshape(final_shapes_src, (68,2)).astype('int32')
        final_shapes_dst = np.reshape(final_shapes_dst, (68,2)).astype('int32')

        src_hull = cv2.convexHull(final_shapes_src, False)
        dst_hull = cv2.convexHull(final_shapes_dst, False)
        print(src_hull.shape)


        src_mask = np.zeros_like(img_src)
        src_mask = cv2.fillPoly(src_mask, [src_hull], color =(255,255,255))
        dst_mask = np.zeros_like(img_dst)
        dst_mask = cv2.fillPoly(dst_mask, [dst_hull], color =(255,255,255))

        params_x, params_y, M = get_TPS_params(final_shapes_src, final_shapes_dst)
        # print(params_x)
        print('shapes: ',np.shape(params_x),np.shape(params_y))
        # print(params_y)
        
        locs,interior_pts  = warp_TPS(final_shapes_src,final_shapes_dst,img_src,img_dst,src_hull,dst_hull,params_x,params_y,M)
        locs = locs.astype(np.int32)
        print(locs[:10])
        interior_pts = interior_pts.astype(np.int32)
        warp_TPS_img = copyPixels(x_source=locs[:,0],y_source=locs[:,1],x_target=interior_pts[:,0],y_target=interior_pts[:,1],\
            img_src=img_dst,img_dst=img_src)

        # cv2.imshow('warp_TPS:',warp_TPS_img)
        # cv2.waitKey(0)
        r = cv2.boundingRect(src_hull)
        center = (r[0]+(r[2]//2), r[1]+(r[3]//2))

        final_img = cv2.seamlessClone(np.uint8(warp_TPS_img), img_src_original, src_mask, center, cv2.NORMAL_CLONE)
        cv2.imshow('blended_img:',final_img)

        cv2.waitKey(0)

if __name__=="__main__": 
    main()