import os
import sys
import cv2
import argparse
import torch
import warnings
import numpy as np
import json
import IPython
from os.path import exists, join, basename
from keras.models import load_model
from collections import namedtuple
from PIL import Image, ImageDraw
from enum import IntEnum, auto
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
from scipy.stats import linregress, norm

from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.utils import get_yolo_boxes
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.utils.log import get_logger

config_path  = "configs/config_giaothong.json"
with open(config_path) as config_buffer:    
    config = json.load(config_buffer)

net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
scale_num = 2
obj_thresh, nms_thresh = 0.5, 0.6
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
infer_model = load_model(config['train']['saved_weights_name'])

def main_tracking(no_cam, max_dist, max_nms, max_iou, max_age, path, save_path, is_show_log):
    def sorter(item):
        id = item[1]
        frame = item[0]
        return int(id),int(frame)
    def sort_(u):
        u = list(u)
        u = sorted(u,key=sorter)
        count = 1
        for i in range (len(u)-1):
            if int(u[i][1]) == int(u[i+1][1]):
                u[i][1] = count
                if i == int((len(u)-2)):
                    u[i+1][1] = count
            elif int(u[i][1]) != int(u[i+1][1]):
                u[i][1] = count
                count = count +1
                if i == int((len(u)-2)):
                    u[i+1][1] = count
        return u

    def convert_order(result, bus, car, moto, truck):
        for v in result:
            v[2] = float(v[2]) + (float(v[4])/2)
            v[3] = float(v[3]) + (float(v[5])/2)
            v[4] = np.sqrt(float(v[4])*float(v[4]) + float(v[5])*float(v[5]))
            v[5] = -1
            if v[6] == 0:
                v[6] = 3
                bus.append(v)
            elif v[6] == 1:
                v[6] = 2
                car.append(v)
            elif v[6] == 2:
                v[6] = 1
                moto.append(v)
            elif v[6] == 3:
                v[6] = 4
                truck.append(v)
    
    def BoundingBox2xywh(bboxes):
        new_bboxes = []
        for box in bboxes:
            x1 = box.xmin
            y1 = box.ymin
            x2 = box.xmax
            y2 = box.ymax
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            x_c = x1 + w / 2
            y_c = y1 + h / 2
            new_bboxes.append([x_c,y_c,w,h])
        return new_bboxes

    def write_result(result):
        frame_id, tlwhs, track_ids, labels = result
        result_ = []
        for tlwh, track_id, label in zip(tlwhs, track_ids, labels):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            result_.append([frame_id, track_id, x1, y1, w, h, label])
        return result_

    class VideoTracker(object):
        def __init__(self, cfg, use_cuda, cam, save_path, video_path):
            self.cfg = cfg
            self.video_path = video_path
            self.save_path = save_path
            self.use_cuda = use_cuda
            self.cam = cam
            self.logger = get_logger("root")

            use_cuda = self.use_cuda and torch.cuda.is_available()
            if not use_cuda:
                warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

            self.vdo = cv2.VideoCapture()
            self.deepsort = build_tracker(cfg, max_dist, max_nms, max_iou, max_age, use_cuda=use_cuda)

        def __enter__(self):
            if self.cam != -1:
                ret, frame = self.vdo.read()
                assert ret, "Error: Camera error"
                self.im_width = frame.shape[0]
                self.im_height = frame.shape[1]
            else:
                assert os.path.isfile(self.video_path), "Path error"
                self.vdo.open(self.video_path)
                self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
                assert self.vdo.isOpened()

            if is_show_log:
                # logging
                self.logger.info("Save results to {}".format(self.save_path))
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type:
                print(exc_type, exc_value, exc_traceback)

        def run(self):
            bus = []
            truck = []
            car = []
            moto = []
            labels = config['model']['labels']
            idx_frame = 0
            while self.vdo.grab():
                idx_frame += 1
                if idx_frame % 1:
                    continue

                _, ori_im = self.vdo.retrieve()
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh = get_yolo_boxes(infer_model, [im], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
                cls_ids = np.array([box.classes for box in bbox_xywh])
                cls_conf = np.array([box.get_score() for box in bbox_xywh])
                bbox_xywh = np.array(BoundingBox2xywh(bbox_xywh))

                if len(bbox_xywh) > 0:
                    label = []
                    new_bbox = []
                    new_conf = []
                    for it in range(len(cls_ids)):
                        if np.sum(cls_ids[it]) != 0:
                            label.append(np.argmax(cls_ids[it], axis=0))
                            if label[-1] == 2:
                                bbox_xywh[it][-2] *= scale_num
                                bbox_xywh[it][-1] *= scale_num
                            new_bbox.append(bbox_xywh[it])
                            new_conf.append(cls_conf[it])
                    # do tracking
                    new_bbox = np.array(new_bbox)
                    new_conf = np.array(new_conf)
                    if new_bbox.shape != (0,):
                        outputs = self.deepsort.update(new_bbox, new_conf, im, label)

                        if len(outputs) > 0:
                            bbox_tlwh = []
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -2]
                            label = outputs[:, -1]

                            for bb_xyxy in bbox_xyxy:
                                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                            result = [idx_frame - 1, bbox_tlwh, identities, label]
                            result = write_result(result)
                            convert_order(result, bus, car, moto, truck)
                # logging
                if is_show_log:
                    self.logger.info("cam: {}, frame: {}" \
                                    .format(no_cam, idx_frame))
            bus = sort_(bus)
            car = sort_(car)
            moto = sort_(moto)
            truck = sort_(truck)
            return bus, car, moto, truck
    
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    with VideoTracker(cfg=cfg, use_cuda=True, cam=-1, save_path=save_path, \
                      video_path= path + "/cam_" + no_cam + ".mp4") as vdo_trk:
        bus, car, moto, truck = vdo_trk.run()
    
    
    return bus, car, moto, truck
    
TrackItem = namedtuple('TrackItem', ['frame_id', 'obj_type', 'data'])

Event = namedtuple('Event', [
    'video_id', 'frame_id', 'movement_id', 'obj_type', 'confidence', 'track_id',
    'track'])

class ObjectType(IntEnum):

    loai_1 = 1
    loai_2 = 2
    loai_3 = 3
    loai_4 = 4
#Config
IMG_HEIGHT = 720 
IMG_WIDTH = 1280 
FPS = 10
trackid_end =0

def getTrackItemByTrackId(track_id,vehicle_arr):
    track_items = []
    global trackid_end
    if len(vehicle_arr) > 0:
        arr = (vehicle_arr[-1])
        trackid_end =  arr[1]
        for v in vehicle_arr:
            if int(v[1])==track_id:
                class_id = 1 if float(v[6]) == -1.0 else int(float(v[6]))
                confi = 1 if float(v[5]) == -1.0 else float(v[5])
                track_item = TrackItem(int(v[0]), class_id, (float(v[2]), float(v[3]), float(v[4]), confi))
                track_items.append(track_item)
        return track_items
    else:
        return None

def get_region_mask(region, height, width):
    img = Image.new('L', (width, height), 0)
    region = region.flatten().tolist()
    ImageDraw.Draw(img).polygon(region, outline=0, fill=255)
    mask = np.array(img).astype(np.bool)
    return mask

def getROI(path_roi):
  with open(path_roi, "r") as read_file:
      data = json.load(read_file)
      d = data["shapes"][0]["points"]
      d = np.array(d)
      return d

def get_track(track_items, roi_path,min_length = 0.3,stride=1, gaussian_std = 0.3,speed_window=1,min_speed=10):
    img_height= IMG_HEIGHT
    img_width= IMG_WIDTH
    fps= 10
    min_length = max(3, fps*min_length)
    speed_window = int(speed_window * fps // 2) * 2
    init_frame_id = track_items[0].frame_id
    length = track_items[-1].frame_id - init_frame_id + 1
    if length < min_length:
        return None
    if len(track_items) == length:
        interpolated_track = np.stack([t.data for t in track_items])
    else:
        interpolated_track = np.empty((length, len(track_items[0].data)))
        interpolated_track[:, 0] = -1
        for t in track_items:
            interpolated_track[t.frame_id - init_frame_id] = t.data
        for frame_i, state in enumerate(interpolated_track):
            if int(state[0]) >= 0:
                continue
            for left in range(frame_i - 1, -1, -1):
                if interpolated_track[left, 0] >= 0:
                    left_state = interpolated_track[left]
                    break
            for right in range(frame_i + 1, interpolated_track.shape[0]):
                if interpolated_track[right, 0] >= 0:
                    right_state = interpolated_track[right]
                    break
            movement = right_state - left_state
            ratio = (frame_i - left) / (right - left)
            interpolated_track[frame_i] = left_state + ratio * movement
    if gaussian_std is not None:
        gaussian_std = gaussian_std * fps
        track = gaussian_filter1d(
            interpolated_track, gaussian_std, axis=0, mode='nearest')
    else:
        track = interpolated_track
    track = np.hstack([track, np.arange(
        init_frame_id, init_frame_id + length)[:, None]])

    speed_window = min(speed_window, track.shape[0] - 1)
    speed_window_half = speed_window // 2
    speed_window = speed_window_half * 2
    speed = np.linalg.norm(
        track[speed_window:, :2] - track[:-speed_window, :2], axis=1)
    speed_mask = np.zeros((track.shape[0]), dtype=np.bool)
    speed_mask[speed_window_half:-speed_window_half] = \
        speed >= min_speed
    speed_mask[:speed_window_half] = speed_mask[speed_window_half]
    speed_mask[-speed_window_half:] = speed_mask[-speed_window_half - 1]
    track = track[speed_mask]
    
    k=[]
    for i in range(len(track)):
        if int(track[i][0]) > 1278 or int(track[i][1] > 718):
            k.append(i)
    track = np.delete(track, k,axis=0) 
    track_int = track[:, :2].round().astype(int)
    
    # region = np.loadtxt(roi_path, delimiter=',',dtype=np.int)f
    region = getROI(roi_path)
    region_mask = get_region_mask(region, img_height, img_width)
    iou_mask = region_mask[track_int[:, 1], track_int[:, 0]]
    track = track[iou_mask]
    if track.shape[0] < 1:
        return None
    return track

def getMOI(moi_path):
    with open(os.path.join(moi_path)) as f:
        data = json.load(f)
        if len(data["shapes"])==13:
            data['shapes'][1]['label']="1"
            data['shapes'][2]['label']="2"
            data['shapes'][3]['label']="3"
            data['shapes'][4]['label']="4"
            data['shapes'][5]['label']="5"
            data['shapes'][6]['label']="6"
            data['shapes'][7]['label']="7"
            data['shapes'][8]['label']="8"
            data['shapes'][9]['label']="9"
            data['shapes'][10]['label']="10"
            data['shapes'][11]['label']="11"
            data['shapes'][12]['label']="12"
            data['shapes'].pop(0)
        elif len(data["shapes"]) ==3:
            data['shapes'][1]['label']="1"
            data['shapes'][2]['label']="2"
            data['shapes'].pop(0)
        elif len(data["shapes"]) ==2:
            data['shapes'][1]['label']="1"
            data['shapes'].pop(0)
        elif len(data["shapes"]) ==4:
            data['shapes'][1]['label']="1"
            data['shapes'][2]['label']="2"
            data['shapes'][3]['label']="3"
            data['shapes'].pop(0)
        elif len(data["shapes"]) ==7:
            data['shapes'][1]['label']="1"
            data['shapes'][2]['label']="2"
            data['shapes'][3]['label']="3"
            data['shapes'][4]['label']="4"
            data['shapes'][5]['label']="5"
            data['shapes'][6]['label']="6"
            data['shapes'].pop(0)
        return data,len(data["shapes"])

def get_movement_heatmaps(movements, height, width):
    distance_heatmaps = np.empty((len(movements), height, width))
    proportion_heatmaps = np.empty((len(movements), height, width))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([xs.flatten(), ys.flatten()], axis=1)
    for label, movement_vertices in movements.items():
        vectors = movement_vertices[1:] - movement_vertices[:-1]
        lengths = np.linalg.norm(vectors, axis=-1) + 1e-4
        rel_lengths = lengths / lengths.sum()
        vertex_proportions = np.cumsum(rel_lengths)
        vertex_proportions = np.concatenate([[0], vertex_proportions[:-1]])
        offsets = ((points[:, None] - movement_vertices[None, :-1])
                   * vectors[None]).sum(axis=2)
        fractions = np.clip(offsets / (lengths ** 2), 0, 1)
        targets = movement_vertices[:-1] + fractions[:, :, None] * vectors
        distances = np.linalg.norm(points[:, None] - targets, axis=2)
        nearest_segment_ids = distances.argmin(axis=1)
        nearest_segment_fractions = fractions[
            np.arange(fractions.shape[0]), nearest_segment_ids]
        distance_heatmap = distances.min(axis=1)
        proportion_heatmap = vertex_proportions[nearest_segment_ids] + \
            rel_lengths[nearest_segment_ids] * nearest_segment_fractions
        distance_heatmaps[label - 1, ys, xs] = distance_heatmap.reshape(
            height, width)
        proportion_heatmaps[label - 1, ys, xs] = proportion_heatmap.reshape(
            height, width)
    return distance_heatmaps, proportion_heatmaps

def get_movement_scores(track, obj_type, moi_path, proportion_thres_to_delta=0.5, distance_base_size=4,
                        distance_scale=5, start_period=0.3, start_thres=0.5,proportion_scale=0.8,distance_slope_scale=2,merge_detection_score=False,final=True):
    img_height= IMG_HEIGHT
    img_width= IMG_WIDTH
    fps=FPS
    data,dem=getMOI(moi_path)
    movements = {int(shape['label']): np.array(shape['points']) for shape in data['shapes']}
    assert len(movements) == max(movements.keys())
    positions = track[:, :2].round().astype(int)
    
    diagonals = track[:, 2]
    detection_scores = track[:, 3]
    frame_ids = track[:, -1]

    distance_heatmaps, proportion_heatmaps =  get_movement_heatmaps(movements, img_height, img_width)
    distances = distance_heatmaps[:, positions[:, 1], positions[:, 0]]
    proportions = proportion_heatmaps[:, positions[:, 1], positions[:, 0]]

    distances = distances / diagonals[None]
    mean_distances = distances.mean(axis=1)

    x = np.linspace(0, 1, proportions.shape[1])
    distance_slopes = np.empty((len(movements)))
    proportion_slopes = np.empty((len(movements)))
    for movement_i in range(len(movements)):
        distance_slopes[movement_i] = linregress(
            x, distances[movement_i])[0]
        proportion_slopes[movement_i] = linregress(
            x, proportions[movement_i])[0]
    
    proportion_delta = proportions.max(axis=1) - proportions.min(axis=1)
    proportion_slopes = np.where(
        proportion_slopes >= proportion_thres_to_delta,
        proportion_delta, proportion_slopes)
    if obj_type == ObjectType.loai_3:
        distance_base_scale = min(
            1, distance_base_size / mean_distances.shape[0])
        distance_base = np.sort(mean_distances)[
            :distance_base_size].sum() * distance_base_scale
        score_1 = 1 - (mean_distances / distance_base) ** 2
    elif obj_type == ObjectType.loai_4:
        distance_base_scale = min(
            1, distance_base_size / mean_distances.shape[0])
        distance_base = np.sort(mean_distances)[
            :distance_base_size].sum() * distance_base_scale
        score_1 = 1 - (mean_distances / distance_base) ** 2
    else:
        score_1 = expit(4 - mean_distances * distance_scale)

    proportion_factor = 1 / proportion_scale

    score_2 = proportion_factor * np.minimum(
        proportion_slopes, 1 / (proportion_slopes + 1e-8))
    start_period=start_period*fps
    if frame_ids[0] <= start_period and \
            score_2.max() <= start_thres:
        score_2 *= proportion_factor
    score_3 = norm.pdf(distance_slopes * distance_slope_scale) / 0.4
    scores = np.stack([score_1, score_2, score_3], axis=1)
    if final:
        scores = np.clip(scores, 0, 1).prod(axis=1)
        if merge_detection_score:
            scores = scores * detection_scores.mean()
    return scores

import random
def get_obj_type(track_items, track):
    class_id = track_items[0].obj_type
    if class_id == 1:
        obj_type = ObjectType.loai_1
    elif class_id == 2:
        obj_type = ObjectType.loai_2
    elif class_id == 3:
        obj_type = ObjectType.loai_3
    elif class_id == 4:
        obj_type = ObjectType.loai_4

    return obj_type

def get_event(video_id, track_id, vehicle, roi_path, moi_path, stride=1,min_score=0.000001,return_all_events=False):

    #Get track_id item
    
    track_items = getTrackItemByTrackId(track_id,vehicle)

    if track_items is None:
        return None

    track = get_track(track_items,roi_path)

    if track is None:
        return None
    obj_type = get_obj_type(track_items, track)
    # obj_type = 1111
    frame_id = (track_items[-1][0] + 1) * stride

    #Get movement_scores
    global movement_scores
    movement_scores = get_movement_scores(track, obj_type,moi_path)

    max_index = movement_scores.argmax()
    max_score = movement_scores[max_index]
    if len(movement_scores) >2:  
        if max_score < min_score:
            movement_id = random.randint(1, len(movement_scores))
        else:
            movement_id = max_index + 1
    elif len(movement_scores) ==1:
        movement_id = 1
    elif len(movement_scores) ==2: 
        if movement_scores[0] > movement_scores[1]:
            movement_id = 1
        elif movement_scores[0] < movement_scores[1]:
            movement_id = 2
        else:
            movement_id = random.randint(1, 2)
    event = Event(video_id, frame_id, movement_id, obj_type, max_score, track_id, track_items)
    return event

def get_multi_event(video_id,st_id,en_id,vehicle, roi_path, moi_path):
    events = []
    for track_id in range(st_id,en_id+1):
        event = get_event(video_id,track_id,vehicle, roi_path, moi_path)
        if event is None:
            continue
        events.append(event)
    events.sort(key=lambda x: x.frame_id)
    return events

def create_submission(results,result_file,j, p):
    with open(result_file,'w') as f:
        for d in results:
            f.write("{:s} {} {} {}\n".format("cam_"+str(j),d.frame_id,d.movement_id,d.obj_type.value))

def main_counting(no_cam, path, bus, car, moto, truck):
    vehicles = (bus, car, moto, truck)
    vehicles_ = ("bus", "car", "moto", "truck")
    path_roi= path + "/cam_" + no_cam + ".json"
    data,cd = getMOI(path_roi)
    p = np.zeros(cd*4, dtype=int)
    for vehicle, vehicle_ in zip(vehicles, vehicles_):
        trackitem = getTrackItemByTrackId(1,vehicle)
        video_id = 0
        trackid_st = 1
        trackid_en = int(trackid_end)
        results = get_multi_event(video_id,trackid_st,trackid_en,vehicle,path_roi,path_roi)
        create_submission(results, "process/" + vehicle_ + no_cam +".txt", no_cam, p)

#getresult
def merge_file(vehicle_txt_path, save_path, no_cam):
    t= []
    # Đường dẫn lưu kết quả từ counting
    with open(vehicle_txt_path + "/moto" + no_cam + ".txt", "r",encoding="utf-8") as file1:
        data = file1.read().splitlines()
        for i in range (len(data)):
            v = data[i].split(",")
            t.append(v)
        file1.close()
    with open(vehicle_txt_path + "/car" + no_cam + ".txt", "r",encoding="utf-8") as file2:
        data = file2.read().splitlines()
        for i in range (len(data)):
            v = data[i].split(",")
            t.append(v)
        file2.close()
    with open(vehicle_txt_path + "/bus" + no_cam + ".txt", "r",encoding="utf-8") as file3:
        data = file3.read().splitlines()
        for i in range (len(data)):
            v = data[i].split(",")
            t.append(v)
        file3.close()
    with open(vehicle_txt_path + "/truck" + no_cam + ".txt", "r",encoding="utf-8") as file4:
        data = file4.read().splitlines()
        for i in range (len(data)):
            v = data[i].split(",")
            t.append(v)
        file4.close()
    t = np.array(t)
    with open(save_path + "/total_" + no_cam + ".txt", "w",encoding="utf-8") as f: 
      for i in t:
          i = ', '.join(str(x) for x in i)
          f.write(i+"\n")

def main(no_cam, max_dist, max_nms, max_iou, max_age, json_path, full_vid_path, save_path, is_show_log, x=None):
    bus, car, moto, truck = main_tracking(no_cam, max_dist, max_nms, max_iou, max_age, full_vid_path, save_path, is_show_log)
    #main_convert(no_cam, save_path)
    main_counting(no_cam, json_path, bus, car, moto, truck)
    merge_file("process", save_path, no_cam)

#Run on full-length video
max_dist = 0.15
max_age = 30
def full_vid_main(no_cam, params_path, json_path, save_path, full_vid_path):
    #Get param from file
    with open(params_path + "/param_" + no_cam + ".txt", 'r') as f:
        data = f.read().splitlines()
    if len(data) == 0:
        print("Cannot find " + params_path + "/param_" + no_cam + ".txt")
    else:
        arr = (data[-1])
        max_nms, max_iou = arr.split(",")
        main(no_cam, max_dist, float(max_nms), float(max_iou), max_age, json_path, full_vid_path, save_path, True)

        