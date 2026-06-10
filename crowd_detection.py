"""
================================================================================
GARUDAI: Crowd Activity & Overcrowding Detection Module
================================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
import json
import argparse
import sys

# ================================================================================
# CONFIGURATION
# ================================================================================

CROWD_THRESHOLD = 5
HYSTERESIS_FRAMES = 3
ALERT_COOLDOWN = 60
MODEL_NAME = "yolo11s.pt"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS_ID = 0
DEFAULT_VIDEO_SOURCE = 0
ZONE_CONFIG_FILE = "zone.json"

# ================================================================================
# STATE MACHINE
# ================================================================================

class CrowdState(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"

@dataclass
class CrowdStateManager:
    state: CrowdState = CrowdState.NORMAL
    last_alert_time: datetime = None
    hysteresis_counter: int = 0
    
    STATE_COLORS = {
        CrowdState.NORMAL: (0, 255, 0),
        CrowdState.WARNING: (0, 255, 255),
        CrowdState.CRITICAL: (0, 0, 255)
    }
    
    def update(self, current_count: int, threshold: int):
        if current_count > threshold:
            target_state = CrowdState.CRITICAL
        elif current_count == threshold:
            target_state = CrowdState.WARNING
        else:
            target_state = CrowdState.NORMAL
        
        if target_state != self.state:
            self.hysteresis_counter += 1
            if self.hysteresis_counter >= HYSTERESIS_FRAMES:
                old_state = self.state
                self.state = target_state
                self.hysteresis_counter = 0
                edge_triggered = (target_state == CrowdState.CRITICAL and 
                                old_state != CrowdState.CRITICAL)
                return self.state, edge_triggered
        else:
            self.hysteresis_counter = 0
        
        return self.state, False
    
    def get_color(self):
        return self.STATE_COLORS[self.state]
    
    def should_log_alert(self):
        if self.last_alert_time is None:
            return True
        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed >= ALERT_COOLDOWN
    
    def mark_alert_sent(self):
        self.last_alert_time = datetime.now()

# ================================================================================
# ZONE MAPPER
# ================================================================================

class ZoneMapper:
    def __init__(self, zone_config_path: str):
        self.config_path = zone_config_path
        self.normalized_polygon = None
        self._load_zone_config()
    
    def _load_zone_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            coords = config.get('polygon', [])
            if not coords:
                raise ValueError("No polygon coordinates found")
            is_normalized = all(0 <= pt[0] <= 1 and 0 <= pt[1] <= 1 for pt in coords)
            if is_normalized:
                self.normalized_polygon = np.array(coords, dtype=np.float32)
                print(f"[INFO] Loaded normalized zone coordinates")
            else:
                self.normalized_polygon = np.array([
                    [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]
                ], dtype=np.float32)
                print(f"[INFO] Using default zone")
        except FileNotFoundError:
            self.normalized_polygon = np.array([
                [0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]
            ], dtype=np.float32)
            print(f"[WARN] Using default zone")
    
    def get_polygon_for_frame(self, frame_shape: tuple):
        height, width = frame_shape[:2]
        if self.normalized_polygon is not None:
            scaled_polygon = self.normalized_polygon.copy()
            scaled_polygon[:, 0] *= width
            scaled_polygon[:, 1] *= height
            return scaled_polygon.astype(np.int32)
        return np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.int32)

# ================================================================================
# CROWD DETECTOR
# ================================================================================

class CrowdDetector:
    def __init__(self, model_path: str, zone_config_path: str, threshold: int):
        self.model = YOLO(model_path)
        self.zone_mapper = ZoneMapper(zone_config_path)
        self.threshold = threshold
        self.state_manager = CrowdStateManager()
        self.logger = self._setup_logger()
        self.tracked_ids = set()
        
    def _setup_logger(self):
        logger = logging.getLogger('GarudAI')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
        return logger
    
    @staticmethod
    def get_anchor_point(bbox):
        x1, y1, x2, y2 = bbox
        anchor_x = int((x1 + x2) / 2)
        anchor_y = int(y2)
        return (anchor_x, anchor_y)
    
    def process_frame(self, frame):
        zone_polygon = self.zone_mapper.get_polygon_for_frame(frame.shape)
        
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, classes=[TARGET_CLASS_ID], verbose=False)
        
        count = 0
        boxes_inside_zone = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes[i].xyxy[0].cpu().numpy()
                track_id = boxes[i].id
                anchor = self.get_anchor_point(bbox)
                is_inside = cv2.pointPolygonTest(zone_polygon, anchor, False) >= 0
                
                if is_inside:
                    count += 1
                    if track_id is not None:
                        self.tracked_ids.add(int(track_id))
                    boxes_inside_zone.append({
                        'bbox': bbox,
                        'anchor': anchor,
                        'track_id': int(track_id) if track_id is not None else None
                    })
        
        state, edge_triggered = self.state_manager.update(count, self.threshold)
        
        if edge_triggered:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert_msg = f"[ALERT] Overcrowding Detected at {timestamp}!"
            self.logger.warning(alert_msg)
            self.state_manager.mark_alert_sent()
        
        return {
            'count': count,
            'state': state,
            'color': self.state_manager.get_color(),
            'edge_triggered': edge_triggered,
            'boxes': boxes_inside_zone,
            'zone_polygon': zone_polygon,
            'tracked_count': len(self.tracked_ids)
        }

# ================================================================================
# VISUALIZER
# ================================================================================

class Visualizer:
    @staticmethod
    def draw_zone_polygon(frame, polygon, color, thickness=3):
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame
    
    @staticmethod
    def draw_bounding_boxes(frame, boxes, color):
        for box_info in boxes:
            bbox = box_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            anchor = box_info['anchor']
            cv2.circle(frame, anchor, 5, (0, 0, 255), -1)
            if box_info['track_id'] is not None:
                label = f"ID:{box_info['track_id']}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    @staticmethod
    def draw_status_banner(frame, count, threshold, state, color):
        banner_height = 80
        h, w = frame.shape[:2]
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
        banner[:] = color
        cv2.rectangle(banner, (0, 0), (w, banner_height), (50, 50, 50), 2)
        status_text = f"Crowd Status: {count} / {threshold}"
        cv2.putText(banner, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(banner, f"[{state.value}]", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        frame[0:banner_height, 0:w] = banner
        return frame
    
    @staticmethod
    def draw_info_overlay(frame, fps=0, tracked_total=0):
        h, w = frame.shape[:2]
        info_text = f"Model: YOLO11s | Tracked IDs: {tracked_total}"
        cv2.putText(frame, info_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

# ================================================================================
# MAIN APP
# ================================================================================

class GarudAICrowdApp:
    def __init__(self, video_source=DEFAULT_VIDEO_SOURCE, threshold=CROWD_THRESHOLD):
        self.video_source = video_source
        self.threshold = threshold
        self.detector = None
        self.visualizer = Visualizer()
        self.running = False
        
    def initialize(self):
        print("=" * 60)
        print("GARUDAI: Crowd Activity & Overcrowding Detection")
        print("=" * 60)
        print(f"[CONFIG] Threshold: {self.threshold} people")
        print(f"[CONFIG] Hysteresis: {HYSTERESIS_FRAMES} frames")
        print(f"[CONFIG] Model: {MODEL_NAME}")
        print("=" * 60)
        
        self.detector = CrowdDetector(MODEL_NAME, ZONE_CONFIG_FILE, self.threshold)
        
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print("[INFO] Initialized. Starting detection...\n")
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            result = self.detector.process_frame(frame)
            
            frame = self.visualizer.draw_zone_polygon(frame, result['zone_polygon'], result['color'], 3)
            frame = self.visualizer.draw_bounding_boxes(frame, result['boxes'], result['color'])
            frame = self.visualizer.draw_status_banner(frame, result['count'], self.threshold, result['state'], result['color'])
            current_fps = self.fps if self.fps > 0 else 30
            frame = self.visualizer.draw_info_overlay(frame, current_fps, result['tracked_count'])
            
            cv2.imshow('GarudAI - Crowd Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] Shutting down...")
                break
        
        self.cleanup()
        
    def cleanup(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='GarudAI Crowd Detection')
    parser.add_argument('-v', '--video', type=str, default=str(DEFAULT_VIDEO_SOURCE))
    parser.add_argument('-t', '--threshold', type=int, default=CROWD_THRESHOLD)
    parser.add_argument('-z', '--zone', type=str, default=ZONE_CONFIG_FILE)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    app = GarudAICrowdApp(video_source=video_source, threshold=args.threshold)
    app.initialize()
    app.run()