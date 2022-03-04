import cv2
import numpy as np
import dlib
import random 

def get_facial_landmarks(img):
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./packages/mmod_human_face_detector.dat")
    landmark_detector = dlib.shape_predictor("./packages/shape_predictor_68_face_landmarks.dat")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = dnnFaceDetector(img_gray, 1)
    final_rects = []
    final_shapes = []
    for r in rects:
        two_pts = [(r.rect.left(),r.rect.top()), (r.rect.right(),r.rect.bottom())]
        shape = landmark_detector(img_gray, r.rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        final_shapes.append(shape_np)
        final_rects.append(two_pts)
    return final_rects, final_shapes
# #SIMILAR FUNCTION WITH DIFFERENT FACE DETECTION 
# def get_facial_landmarks(img):
#     dnnFaceDetector = dlib.get_frontal_face_detector()
#     landmark_detector = dlib.shape_predictor("./dlib-models-master/shape_predictor_68_face_landmarks.dat")
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     rects = dnnFaceDetector(img_gray, 1)
#     final_rects = []
#     final_shapes = []
#     for r in rects:
#         two_pts = [(r.left(),r.top()), (r.right(),r.bottom())]
#         shape = landmark_detector(img_gray, r)
#         shape_np = np.zeros((68, 2), dtype="int")
#         for i in range(0, 68):
#             shape_np[i] = (shape.part(i).x, shape.part(i).y)
#         final_shapes.append(shape_np)
#         final_rects.append(two_pts)
#     return final_rects, final_shapes

# Check if a point is inside a rectangle
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

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


## destination -- the celebrity image?
## source -- main image (the image we want to overlay the celebrity image to) 
# read image
dest = cv2.imread("./data/celebrity.jpg")
size = dest.shape 
dest_original = dest.copy()
source = cv2.imread("./data/test2.jpg")
size = source.shape 
source_original = source.copy()
dest_original = dest.copy()
# get face detection and facial landmarks
source_rect, source_shapes = get_facial_landmarks(source)
dest_rect, dest_shapes = get_facial_landmarks(dest)

tempS = source.copy()
tempD = dest.copy()
print("Total number of faces detected: ", len(source_rect)+len(dest_rect))
## Assuming for now that each image has same number of faces
for i in range(len(source_rect)):
    r = source_rect[i]
    start_point = r[0]
    end_point = r[1]
    tempS = cv2.rectangle(tempS, start_point, end_point, (0,255,0), thickness=2)
    r = dest_rect[i]
    start_point = r[0]
    end_point = r[1]
    tempD = cv2.rectangle(tempD, start_point, end_point, (0,255,0), thickness=2)
    for j, (x, y) in enumerate(source_shapes[i]):
        # Draw the circle to mark the keypoint 
        cv2.circle(tempS, (x, y), 1, (0, 0, 255), 2)
    for j, (x, y) in enumerate(dest_shapes[i]):
        # Draw the circle to mark the keypoint 
        cv2.circle(tempD, (x, y), 1, (0, 0, 255), 2)
cv2.imshow('face detection source', tempS)
cv2.imshow('face detection destination', tempD)
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()

# reshape as points for triangulation
old_shape = np.shape(source_shapes)
source_shapes_new = np.reshape(source_shapes, (old_shape[0]*old_shape[1], old_shape[2]))
source_size = (0, 0, size[1], size[0])
subdiv  = cv2.Subdiv2D(source_size)
for p in source_shapes_new:
    p = (p[0],p[1])
    subdiv.insert(p)
draw_delaunay(source, subdiv, (255, 255, 255))
old_shape = np.shape(dest_shapes)
dest_shapes_new = np.reshape(dest_shapes, (old_shape[0]*old_shape[1], old_shape[2]))
dest_size = (0, 0, size[1], size[0])
subdiv  = cv2.Subdiv2D(dest_size)
for p in dest_shapes_new:
    p = (p[0],p[1])
    subdiv.insert(p)
draw_delaunay(dest, subdiv, (255, 255, 255))


cv2.imshow("Delaunay Triangulation", source)
cv2.imshow("Delaunay Triangulation", dest)
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()