
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
from tqdm import tqdm
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, get_3D_bbox

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer_multiple import Renderer
import config
import constants

from collections import namedtuple
from posetrack_json import *



if __name__ == "__main__":
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_root = "/project/lighttrack/data/Data_2018/posetrack_data"
    detections_openSVAI_folder = osp.join("/project/lighttrack/DeformConv_FPN_RCNN_detect_CPN_format")
    det_file_paths = get_immediate_childfile_paths(detections_openSVAI_folder)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)


    for video_idx, det_file_path in tqdm(enumerate(det_file_paths)):
        if video_idx != 5:
             continue
        precomputed_dets = load_det_from_CPNformat(det_file_path) # Get frames

        pred_betas = []
        pred_rotmat = []
        pred_camera = []
        bbox_3d = []
        bbox_2d = []
        if precomputed_dets[1] == []:
            continue
        for det in precomputed_dets[1]:
            pred_betas.append(torch.FloatTensor(np.array(det['pred_betas'])).to(device))
            pred_rotmat.append(torch.FloatTensor(np.array(det['pred_rotmat'])).to(device))
            pred_camera.append(torch.FloatTensor(np.array(det['pred_camera'])).to(device))
            bbox_3d.append(np.array(det['bbox_3d'])[None,...])
            bbox_2d.append(np.array(det['bbox'])[None,...])
        pred_betas = torch.cat(pred_betas,dim=0)
        pred_rotmat = torch.cat(pred_rotmat,dim=0)
        pred_camera = torch.cat(pred_camera,dim=0)
        bbox_3d = np.concatenate(bbox_3d,axis=0)
        bbox_2d = np.concatenate(bbox_2d,axis=0)

        with torch.no_grad():
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices



        img_path = os.path.join(precomputed_dets[1][0]["imgpath"]).replace('/export/guanghan/','/project/lighttrack/data/')
        img = cv2.imread(img_path)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
        IMG_RES = max(img.shape[0],img.shape[1])
        # Calculate camera parameters for rendering
        pred_vertices = pred_vertices.cpu().numpy()
        
        xyz = []
        for n in range(pred_vertices.shape[0]):
            bbox = bbox_2d[n]
            resolution = max(bbox[2],bbox[3])
            pred_cam_t = torch.stack([pred_camera[n,1],
                                      pred_camera[n,2],
                                      2*constants.FOCAL_LENGTH/(resolution * pred_camera[n,0] +1e-9)],dim=-1)
            pred_cam_t[0] += (bbox[0]+0.5*bbox[2])*pred_cam_t[2]/constants.FOCAL_LENGTH # pixel * Z / f
            pred_cam_t[1] += (bbox[1]+0.5*bbox[3])*pred_cam_t[2]/constants.FOCAL_LENGTH # pixel * Z / f
            xyz.append(pred_cam_t[None,:])
        mesh_trans = torch.cat(xyz,dim=0).cpu().numpy()
        renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=IMG_RES, faces=smpl.faces,img=img)
        img_shape = renderer(pred_vertices, np.ones_like(img) ,mesh_trans)

        cv2.imwrite('frame_%d_debug.png' %video_idx, 255 * img_shape[:,:,::-1])
