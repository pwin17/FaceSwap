#!/usr/bin/python3
import cv2
import numpy as np
from api import PRN
from utils.render import render_texture
import os

# global prnet


def prnet_one_face_main(prn, img_src, img_dst):
    print("processing source image")
    img_src =cv2.resize(img_src,(500,500))
    img_src_original = img_src.copy()
    [src_h, src_w, _] = img_src.shape

    
    pos_src = prn.process(img_src)
    vertices_src = prn.get_vertices(pos_src)
    img_src = img_src/255.0
    texture_src = cv2.remap(img_src, pos_src[:,:,:2].astype(np.float32), None,\
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    new_texture = texture_src

    print("processing destination image")

    img_dst =cv2.resize(img_dst,(500,500))
    img_dst_original = img_dst.copy()
    [dst_h, dst_w, _] = img_dst.shape
    pos_dst = prn.process(img_dst)
    vertices_dst = prn.get_vertices(pos_dst)
    img_dst = img_dst/255.0
    texture_dst = cv2.remap(img_dst, pos_dst[:,:,:2].astype(np.float32), None, \
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # new_texture = texture_dst
    print("Swapping")

    vis_colors = np.ones((vertices_dst.shape[0], 1))
    face_mask = render_texture(vertices_dst.T, vis_colors.T, prn.triangles.T, dst_h, dst_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_dst.T, new_colors.T, prn.triangles.T, dst_h, dst_w, c=3)
    new_image = img_dst*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (img_dst*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    return output 

def prnet_two_faces_main(prn, img_src, img_dst, idx1, idx2):
    print("processing source image")

    [ori_src_h, ori_src_w, _] = img_src.shape
    img_src =cv2.resize(img_src,(ori_src_w//2, ori_src_h//2))
    [src_h, src_w, _] = img_src.shape

    pos_src = prn.process(img_src, index=idx1)
    vertices_src = prn.get_vertices(pos_src)
    img_src = img_src/255.0
    texture_src = cv2.remap(img_src, pos_src[:,:,:2].astype(np.float32), None,\
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    new_texture = texture_src

    print("processing destination image")

    [dst_h, dst_w, _] = img_dst.shape
    img_dst =cv2.resize(img_dst,(dst_w//2,dst_h//2))
    [dst_h, dst_w, _] = img_dst.shape
    pos_dst = prn.process(img_dst, index=idx2)
    vertices_dst = prn.get_vertices(pos_dst)
    img_dst = img_dst/255.0
    texture_dst = cv2.remap(img_dst, pos_dst[:,:,:2].astype(np.float32), None, \
                            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    # img_out = img_dst.copy()
    # img_out = img_out / 255.0
    
    print("Swapping")
    # Swapping
    vis_colors = np.ones((vertices_dst.shape[0], 1))
    face_mask = render_texture(vertices_dst.T, vis_colors.T, prn.triangles.T, dst_h, dst_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_dst.T, new_colors.T, prn.triangles.T, dst_h, dst_w, c=3)

    img_out = img_dst*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]
    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    img_out = cv2.seamlessClone((img_out*255).astype(np.uint8), (img_dst*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    img_out = img_out/255.0
    img_out_copy = img_out.copy()

    # Swapping
    print("Swapping")
    new_texture = texture_dst
    vis_colors = np.ones((vertices_src.shape[0], 1))
    face_mask = render_texture(vertices_src.T, vis_colors.T, prn.triangles.T, src_h, src_w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices_src.T, new_colors.T, prn.triangles.T, src_h, src_w, c=3)
    img_out = img_out_copy*(1- face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    print("Blending")
    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((img_out*255).astype(np.uint8), (img_out_copy*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    output = cv2.resize(output, (ori_src_w, ori_src_h), interpolation=cv2.INTER_LINEAR)

    return output
if __name__=="__main__": 

    img_src = cv2.imread('../data/put-zel.jpg')
    img_dst = cv2.imread('../data/put-zel.jpg')
 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    prnet = PRN(is_dlib=True)
    print("---------------------")
    output_img1 = prnet_two_faces_main(prnet, img_src, img_dst, 0, 1)


    img_src = cv2.imread('../data/celebrity.jpg')
    img_dst = cv2.imread('../data/ron.jpg')

    output_img2 = prnet_one_face_main(prnet, img_src, img_dst)

    cv2.imshow("one face",output_img2)
    cv2.imshow("two faces", output_img1)
    cv2.waitKey(0)