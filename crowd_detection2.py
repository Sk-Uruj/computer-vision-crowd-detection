import cv2
import numpy as np
import argparse
import json
import time
import os
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZoneMapper:
    """Resolution-independent zone coordinate mapper."""
    
    def __init__(self, zone_config: Dict[str, Any]):
        self.zone_points = self._extract_zone_points(zone_config)
    
    def _extract_zone_points(self, zone_config: Dict[str, Any]) -> List[List[float]]:
        """Extract zone points from various JSON formats."""
        point_keys = ['points', 'zone_points', 'coordinates']
        points = None
        
        for key in point_keys:
            if key in zone_config:
                points = zone_config[key]
                break
        
        if points is None:
            raise ValueError(f"Zone config missing: {point_keys}")
        
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("Zone needs ≥3 points")
        
        for i, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"Point {i}: [x_norm, y_norm] expected")
            x, y = point
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Point {i}: {point} (use 0.0-1.0)")
        
        logger.info(f"✅ Zone loaded: {len(points)} points")
        return points
    
    def map_to_frame(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        mapped_points = [[int(x * w), int(y * h)] for x, y in self.zone_points]
        return np.array(mapped_points, dtype=np.int32)

class CrowdDetector:
    CROWD_THRESHOLD = 5
    CONFIDENCE_THRESHOLD = 0.25
    DISPLAY_WIDTH = 960
    DISPLAY_HEIGHT = 540
    HYSTERESIS_FRAMES = 3
    WAIT_KEY_DELAY = 30
    
    STATE_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
    STATE_LABELS = {0: 'SAFE 🟢', 1: 'WARNING 🟡', 2: 'ALERT 🔴'}
    
    def __init__(self, video_source: str, zone_file: str = 'zone.json'):
        self.video_source = video_source
        self.zone_file = zone_file
        
        # Load zone (auto-create if missing)
        self.zone_config = self._load_zone_config()
        self.zone_mapper = ZoneMapper(self.zone_config)
        
        # State tracking
        self.frame_count_history = deque(maxlen=self.HYSTERESIS_FRAMES)
        self.overcrowd_alert_latch = False
        self.current_state = 0
        self.cap = None
        self.model = None
        
        self._initialize_system()
    
    def _load_zone_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.zone_file):
            logger.warning(f"📍 '{self.zone_file}' missing → creating default")
            self._create_default_zone()
        
        try:
            with open(self.zone_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Invalid JSON in '{self.zone_file}': {e}")
    
    def _create_default_zone(self):
        default_zone = {
            "name": "Default Campus Zone",
            "points": [[0.1,0.7], [0.9,0.7], [0.85,0.45], [0.15,0.45]]
        }
        with open(self.zone_file, 'w') as f:
            json.dump(default_zone, f, indent=2)
        logger.info(f"✅ Created '{self.zone_file}'")
    
    def _initialize_system(self):
        """Robust YOLO initialization with fallbacks."""
        # Video
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Video: {self.video_source}")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"📹 {w}x{h} @ {fps:.1f}fps")
        
        # YOLO with smart fallbacks
        from ultralytics import YOLO
        candidates = ['./yolov11s.pt', 'yolo11s.pt', './yolov8s.pt', 'yolov8s.pt', 'yolov8n.pt']
        
        for model_path in candidates:
            try:
                logger.info(f"🔍 {model_path}")
                self.model = YOLO(model_path)
                # Quick test
                _ = self.model.track(np.zeros((320,320,3), dtype=np.uint8), verbose=False)
                logger.info(f"✅ {model_path}")
                break
            except Exception:
                continue
        
        if not self.model:
            raise RuntimeError("❌ No YOLO model works")
    
    def _get_bottom_center(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return (x1 + x2) // 2, y2
    
    def _count_people_in_zone(self, results, frame_shape):
        zone_pts = self.zone_mapper.map_to_frame(frame_shape)
        count = 0
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if int(box.cls) == 0 and box.conf >= self.CONFIDENCE_THRESHOLD:
                        foot = self._get_bottom_center(box.xyxy[0].cpu().numpy())
                        if cv2.pointPolygonTest(zone_pts, foot, False) >= 0:
                            count += 1
        return count
    
    def _update_state(self, count):
        self.frame_count_history.append(count)
        if len(self.frame_count_history) < self.HYSTERESIS_FRAMES:
            return
        
        avg = sum(self.frame_count_history) / self.HYSTERESIS_FRAMES
        new_state = 2 if avg > self.CROWD_THRESHOLD else 1 if avg == self.CROWD_THRESHOLD else 0
        
        if new_state == 2 and self.current_state != 2:
            self._trigger_alert()
        self.current_state = new_state
    
    def _trigger_alert(self):
        if not self.overcrowd_alert_latch:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.warning(f"🚨 OVERCROWDING! {self.frame_count_history[-1]}/{self.CROWD_THRESHOLD} @ {ts}")
            self.overcrowd_alert_latch = True
    
    def _reset_alert(self):
        if self.current_state != 2:
            self.overcrowd_alert_latch = False
    
    def _draw_zone(self, frame):
        overlay = frame.copy()
        pts = self.zone_mapper.map_to_frame(frame.shape)
        color = self.STATE_COLORS[self.current_state]
        
        cv2.polylines(overlay, [pts], True, color, 3)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame
    
    def _draw_banner(self, frame):
        h, w = frame.shape[:2]
        color = self.STATE_COLORS[self.current_state]
        count = self.frame_count_history[-1] if self.frame_count_history else 0
        
        banner_h = 70
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner[:] = color
        
        status = f"👥 {int(count)}/{self.CROWD_THRESHOLD}"
        state = self.STATE_LABELS[self.current_state]
        
        cv2.putText(banner, status, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)
        cv2.putText(banner, state, (w-280, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)
        
        frame[:banner_h] = banner
        return frame
    
    def _draw_people(self, results, frame):
        pts = self.zone_mapper.map_to_frame(frame.shape)
        color = self.STATE_COLORS[self.current_state]
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if int(box.cls) == 0 and box.conf >= self.CONFIDENCE_THRESHOLD:
                        bbox = box.xyxy[0].cpu().numpy()
                        foot = self._get_bottom_center(bbox)
                        
                        if cv2.pointPolygonTest(pts, foot, False) >= 0:
                            x1, y1, x2, y2 = map(int, bbox)
                            tid = int(box.id[0]) if box.id is not None else 0
                            
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                            cv2.circle(frame, foot, 10, color, -1)
                            cv2.putText(frame, f"ID:{tid}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def run(self):
        logger.info("🚀 GarudAI Crowd Detection v2.0")
        cv2.namedWindow('GarudAI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GarudAI', self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                results = self.model.track(frame, persist=True, conf=self.CONFIDENCE_THRESHOLD, 
                                         classes=[0], verbose=False)
                
                count = self._count_people_in_zone(results, frame.shape)
                self._update_state(count)
                self._reset_alert()
                
                frame = self._draw_zone(frame)
                self._draw_people(results, frame)
                frame = self._draw_banner(frame)
                frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
                
                cv2.imshow('GarudAI', frame)
                
                if cv2.waitKey(self.WAIT_KEY_DELAY) in [ord('q'), 27]:
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
    
    def _cleanup(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='GarudAI Crowd Detection')
    parser.add_argument('video', help='MP4 file')
    parser.add_argument('--zone', default='zone.json', help='Zone JSON')
    args = parser.parse_args()
    
    detector = CrowdDetector(args.video, args.zone)
    detector.run()

if __name__ == "__main__":
    main()