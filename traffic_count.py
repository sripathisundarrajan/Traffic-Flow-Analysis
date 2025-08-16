# traffic_count.py
import os
import cv2
import numpy as np
import pandas as pd
import math
import argparse
import subprocess
from ultralytics import YOLO
from collections import defaultdict, deque
import time

# -------------------- Simple SORT implementation --------------------
# This is a compact, minimal SORT-style tracker for demonstration.
# It's not full-featured DeepSORT. For production, use the official SORT repo.
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def iou(bbox1, bbox2):
    # bbox = [x1,y1,x2,y2]
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    w = max(0, x2-x1)
    h = max(0, y2-y1)
    inter = w*h
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    union = area1+area2-inter
    return inter/union if union>0 else 0

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: x, y, s, r, vx, vy, vs
        # measurement: x, y, s, r
        self.kf.F = np.eye(7)
        for i in range(4):
            self.kf.F[i,i+3] = 1.0  # allow velocity propagation
        self.kf.H = np.zeros((4,7))
        self.kf.H[0,0] = 1
        self.kf.H[1,1] = 1
        self.kf.H[2,2] = 1
        self.kf.H[3,3] = 1
        self.kf.R[2:,2:] *= 10.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        cx = (bbox[0]+bbox[2])/2.
        cy = (bbox[1]+bbox[3])/2.
        s = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        r = (bbox[2]-bbox[0])/(bbox[3]-bbox[1]+1e-6)
        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape((4,1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.age = 0
        self.assigned_bbox = bbox

    def update(self, bbox):
        # convert bbox to measurement
        cx = (bbox[0]+bbox[2])/2.
        cy = (bbox[1]+bbox[3])/2.
        s = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        r = (bbox[2]-bbox[0])/(bbox[3]-bbox[1]+1e-6)
        z = np.array([cx, cy, s, r]).reshape((4,1))
        self.kf.update(z)
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.assigned_bbox = bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update>0:
            self.hits = 0
        self.time_since_update += 1
        # convert state to bbox
        cx, cy, s, r = self.kf.x[0,0], self.kf.x[1,0], self.kf.x[2,0], self.kf.x[3,0]
        w = math.sqrt(abs(s*r))+1e-6
        h = s/(w+1e-6)
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        return [x1, y1, x2, y2]

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2,score]
        trks = []
        for t in self.trackers:
            pred = t.predict()
            trks.append(pred)
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)
        # update matched
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx][:4])
        # create new trackers for unmatched detections
        for idx in unmatched_dets:
            t = KalmanBoxTracker(detections[idx][:4])
            self.trackers.append(t)
        # remove dead trackers
        ret = []
        to_del = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hits>=self.min_hits or True):
                bbox = t.assigned_bbox
                ret.append([bbox[0],bbox[1],bbox[2],bbox[3], t.id])
            if t.time_since_update > self.max_age:
                to_del.append(t)
        for d in to_del:
            try: self.trackers.remove(d)
            except: pass
        return ret

    def associate(self, detections, trks):
        if len(trks)==0:
            return [], list(range(len(detections))), []
        iou_matrix = np.zeros((len(detections), len(trks)), dtype=np.float32)
        for d,det in enumerate(detections):
            for t,trk in enumerate(trks):
                iou_matrix[d,t] = iou(det[:4], trk)
        # Hungarian assignment with -iou (maximize iou)
        det_idx, trk_idx = linear_sum_assignment(-iou_matrix)
        matched, unmatched_dets, unmatched_trks = [], [], []
        for d in range(len(detections)):
            if d not in det_idx:
                unmatched_dets.append(d)
        for t in range(len(trks)):
            if t not in trk_idx:
                unmatched_trks.append(t)
        for di, ti in zip(det_idx, trk_idx):
            if iou_matrix[di,ti] < self.iou_threshold:
                unmatched_dets.append(di)
                unmatched_trks.append(ti)
            else:
                matched.append((di,ti))
        return matched, unmatched_dets, unmatched_trks

# -------------------- End SORT --------------------

# Vehicle classes on COCO
VEHICLE_CLASSES = {"car","motorcycle","bus","truck"}  # adjust as necessary

def download_youtube(url, out_path="video.mp4"):
    # requires yt-dlp installed
    if os.path.exists(out_path):
        print(f"[info] {out_path} already exists, skipping download.")
        return out_path
    cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", out_path, url]
    print("[info] Downloading video with yt-dlp...")
    subprocess.run(cmd, check=True)
    return out_path

def define_three_vertical_lanes(frame_w):
    # returns list of (x1,x2) lane ranges dividing width into 3 vertical lanes
    w = frame_w
    lanes = []
    lane_w = w/3
    for i in range(3):
        lanes.append((int(i*lane_w), int((i+1)*lane_w)))
    return lanes

def lane_of_bbox(bbox, lanes):
    # uses centroid x to decide lane
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)/2.
    for i,(a,b) in enumerate(lanes):
        if cx >= a and cx < b:
            return i+1
    return len(lanes)

def main(args):
    video_path = download_youtube(args.url, "input_video.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] FPS={fps}, W={width}, H={height}")

    lanes = define_three_vertical_lanes(width)
    print(f"[info] lanes: {lanes}")

    # load model
    model = YOLO("yolov8n.pt")  # small and fast; change to yolov8m or v5 for accuracy

    out_video_path = args.output_video or "outputs/result_overlay.mp4"
    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
    csv_rows = []
    seen = {}  # track_id -> {'lane':..., 'first_frame':..., 'frames_seen':..., 'last_seen_frame':...}
    frame_idx = 0
    per_lane_counts = defaultdict(int)

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # detect
        results = model(frame, imgsz=640, conf=0.35, classes=None, device=args.device)  # returns list
        dets = []
        if len(results):
            r = results[0]
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # map class id to name
                name = model.model.names.get(cls, str(cls))
                if name in VEHICLE_CLASSES:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    dets.append([x1,y1,x2,y2,conf])
        # track
        tracks = tracker.update(dets)  # returns list of [x1,y1,x2,y2,track_id]
        # draw lanes
        overlay = frame.copy()
        for i,(a,b) in enumerate(lanes):
            cv2.rectangle(overlay, (a,0), (b,height), (0,0,0), -1)
        alpha = 0.05
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        for tr in tracks:
            x1,y1,x2,y2,tid = tr
            tid = int(tid)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            lane = lane_of_bbox([x1,y1,x2,y2], lanes)
            # update seen info
            if tid not in seen:
                seen[tid] = {'lane': lane, 'first_frame': frame_idx, 'frames_seen':1, 'last_seen_frame': frame_idx}
                per_lane_counts[lane] += 1
                # add CSV row now
                timestamp = frame_idx / fps
                csv_rows.append({'vehicle_id': tid, 'lane': lane, 'frame_count': 1, 'timestamp': timestamp})
            else:
                seen[tid]['frames_seen'] += 1
                seen[tid]['last_seen_frame'] = frame_idx
                # update csv entry frames_seen and timestamp
                for r in csv_rows:
                    if r['vehicle_id']==tid:
                        r['frame_count'] = seen[tid]['frames_seen']
                        r['timestamp'] = frame_idx / fps
            # draw bbox & id
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 3, (0,255,0), -1)
            cv2.putText(frame, f"ID:{tid} L:{lane}", (int(x1), int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # draw lane separators and counts
        for i,(a,b) in enumerate(lanes):
            cv2.line(frame, (a,0), (a,height), (255,0,0), 2)
            cv2.line(frame, (b,0), (b,height), (255,0,0), 2)
            cv2.putText(frame, f"Lane {i+1}: {per_lane_counts[i+1]}",
                        (a+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        out_writer.write(frame)

        if args.display:
            cv2.imshow("overlay", frame)
            key = cv2.waitKey(1)
            if key==27:
                break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time()-start_time
    print(f"[info] Processing done in {elapsed:.1f}s")
    # CSV export
    df = pd.DataFrame(csv_rows)
    csv_out = args.csv or "outputs/counts.csv"
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[info] Saved CSV to {csv_out}")
    # print summary
    print("Final counts per lane:")
    for i in range(1,4):
        print(f" Lane {i}: {per_lane_counts[i]}")
    print(f"Overlay video saved to {out_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=MNn9qKG2UFI",
                        help="YouTube video URL")
    parser.add_argument("--output-video", type=str, default="outputs/result_overlay.mp4")
    parser.add_argument("--csv", type=str, default="outputs/counts.csv")
    parser.add_argument("--display", action="store_true", help="show window while processing")
    parser.add_argument("--device", type=str, default="0", help="device for model (0 for cpu)")
    args = parser.parse_args()
    main(args)
