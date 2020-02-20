#%%
import sys
import os.path as osp
import argparse

sys.path.append("/project/lighttrack/utils")
from utils_json import *
from utils_io_folder import *


global bbox_thresh
bbox_thresh = 0.4


def load_gt_dets_mot(json_folder_input_path):
    ''' load all detections in a video by reading json folder'''
    if json_folder_input_path.endswith(".json"):
        json_file_path = json_folder_input_path
        dets_standard = read_json_from_file(json_file_path)
    else:
        dets_standard = batch_read_json(json_folder_input_path)

    # print("Using detection threshold: ", args.bbox_thresh)
    print("Using detection threshold: ", bbox_thresh)
    dets = standard_to_dicts(dets_standard, bbox_thresh = bbox_thresh)

    print("Number of imgs: {}".format(len(dets)))
    return dets

def load_det_from_CPNformat(json_folder_input_path):
    ''' load all detections in a video by reading json folder'''
    if json_folder_input_path.endswith(".json"):
        json_file_path = json_folder_input_path
        dets_CPNformat = read_json_from_file(json_file_path)
    else:
        dets_CPNformat = batch_read_json(json_folder_input_path)

    # print("Using detection threshold: ", args.bbox_thresh)
    # dets = standard_to_dicts(dets_standard, bbox_thresh = args.bbox_thresh)

    print("Number of imgs: {}".format(len(dets_CPNformat)))
    return dets_CPNformat

def batch_read_json(json_folder_path):
    json_paths = get_immediate_childfile_paths(json_folder_path, ext=".json")

    dets = []
    for json_path in json_paths:
        python_data = read_json_from_file(json_path)
        dets.append(python_data)
    return dets


def standard_to_dicts(dets_standard, bbox_thresh = 0):
    # standard detection format to CPN detection format
    num_dets = len(dets_standard)
    dets_CPN_list = []
    for i in range(num_dets):
        det_standard = dets_standard[i]
        num_candidates = len(det_standard['candidates'])

        dets_CPN = []
        for j in range(num_candidates):
            det = {}
            det['image_id'] = det_standard['image']['id']
            det['bbox'] = det_standard['candidates'][j]['det_bbox']
            det['bbox_score'] = det_standard['candidates'][j]['det_score']
            det['imgpath'] = os.path.join(det_standard['image']['folder'], det_standard['image']['name'])
            if det['bbox_score'] >= bbox_thresh:
                dets_CPN.append(det)
        dets_CPN_list.append(dets_CPN)
    return dets_CPN_list
# %%
from collections import namedtuple

# detections_openSVAI_folder = osp.join("/project/lighttrack/data/Data_2018/posetrack_data/annotations_openSVAI/")
# detections_openSVAI_folder = osp.join("/project/lighttrack/data/Data_2018/posetrack_data/DeformConv_FPN_RCNN_detect/")


# # per det_file_path is an video
# for det_file_path in det_file_paths:
#     precomputed_dets = load_gt_dets_mot(det_file_path)  
#     num_imgs = len(precomputed_dets)
#     img_id = -1
#     while img_id < num_imgs-1:
#         img_id += 1
#         gt_data = precomputed_dets[img_id]
#         if gt_data == []:
#             continue
#         else:
#             img_path = gt_data[0]["imgpath"]
#             bbox_list = [det['bbox'] for det in gt_data]

            
# %%
