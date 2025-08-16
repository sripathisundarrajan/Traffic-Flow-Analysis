# Traffic-Flow-Analysis
# Traffic Flow Analysis — Vehicle Counting per Lane

## Overview
This project detects and tracks vehicles in a traffic video, assigns each vehicle to one of three lanes, counts per-lane vehicles, and exports results.

## Requirements
Install dependencies:
pip install -r requirements.txt

## Usage
python traffic_count.py --url "https://www.youtube.com/watch?v=MNn9qKG2UFI" --display

Outputs:
- outputs/result_overlay.mp4  (video with overlays)
- outputs/counts.csv          (CSV with vehicle_id, lane, frame_count, timestamp)

## Notes
- Uses YOLOv8 (ultralytics). For improved accuracy, change `yolov8n.pt` to a larger model.
- For improved tracking robustness, integrate DeepSORT.
