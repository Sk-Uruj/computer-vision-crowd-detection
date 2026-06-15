import cv2
import numpy as np
import argparse
import json
import os
import time
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
        point_keys = ['points', 'polygon']
        points = None
        
        for key in point_keys:
            if key in zone_config:
                points = zone_config[key]
                break
        
        if points is None or not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Invalid zone points in {zone_config.get('name', 'unknown')}")
        
        # Validate normalized coordinates
        for i, point in enumerate(points):
            x, y = point
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"Point {i} not normalized: {point}")
        
        logger.info(f"✅ Patrol zone loaded: {len(points)} points")
        return points
    
    def map_to_frame(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        mapped_points = [[int(x * w), int(y * h)] for x, y in self.zone_points]
        return np.array(mapped_points, dtype=np.int32)

class PatrolComplianceEngine:
    """Guard patrol verification & temporal compliance tracker."""
    
    PATROL_INTERVAL_SECONDS = 60.0  # Demo: 60s (Production: 1800s = 30min)
    CONFIDENCE_THRESHOLD = 0.3
    PERSON_STOP_DURATION = 2.0  # Seconds person must be in zone
    DISPLAY_WIDTH = 960
    DISPLAY_HEIGHT = 540
    WAIT_KEY_DELAY = 30  # Natural playback speed
    
    STATE_COLORS = {
        0: (0, 255, 0),    # PATROLLED (GREEN)
        1: (0, 255, 255),  # PENDING (YELLOW) 
        2: (0, 0, 255)     # BREACH (RED)
    }
    
    STATE_LABELS = {
        0: "PATROLLED ✅",
        1: "PENDING ⏳", 
        2: "BREACH ❌"
    }
    
    def __init__(self, video_source: str, zone_file: str = 'patrol_zone.json'):
        self.video_source = video_source
        self.zone_file = zone_file
        
        # Load patrol checkpoint zone
        self.zone_config = self._load_zone_config()
        self.zone_mapper = ZoneMapper(self.zone_config)
        
        # Compliance tracking state
        self.last_patrol_timestamp = 0.0
        self.alert_latch = False
        self.current_state = 1  # Start as PENDING
        
        # Person dwell time tracking (track ID → enter time)
        self.person_dwell_times = {}
        self.person_stop_timer = 0.0
        
        # Video & model
        self.cap = None
        self.model = None
        self.fps = 0.0
        self.frame_pos = 0
        
        self._initialize_system()
    
    def _load_zone_config(self) -> Dict[str, Any]:
        """Load patrol zone with auto-creation."""
        if not os.path.exists(self.zone_file):
            logger.warning(f"📍 '{self.zone_file}' missing → creating default")
            self._create_default_patrol_zone()
        
        with open(self.zone_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"✅ Loaded patrol zone: {config.get('name', 'Unnamed')}")
        return config
    
    def _create_default_patrol_zone(self):
        """Create default patrol checkpoint zone."""
        default_zone = {
            "name": "Patrol Checkpoint Alpha",
            "description": "Default bottom-center patrol point",
            "points": [[0.4, 0.75], [0.6, 0.75], [0.6, 0.65], [0.5, 0.60], [0.4, 0.65]],
            "polygon": [[0.4, 0.75], [0.6, 0.75], [0.6, 0.65], [0.5, 0.60], [0.4, 0.65]]
        }
        with open(self.zone_file, 'w') as f:
            json.dump(default_zone, f, indent=2)
    
    def _initialize_system(self):
        """Initialize video capture & YOLO tracker."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open: {self.video_source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0  # Fallback
        logger.info(f"📹 Video: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {self.fps:.1f}fps")
        
        # Robust YOLO initialization
        from ultralytics import YOLO
        model_paths = ['./yolo11s.pt', 'yolo11s.pt', 'yolov8s.pt']
        for path in model_paths:
            try:
                self.model = YOLO(path)
                logger.info(f"✅ YOLO tracker: {path}")
                break
            except:
                continue
        else:
            raise RuntimeError("❌ No YOLO model available")
    
    def _get_current_video_time(self) -> float:
        """Calculate precise video timestamp from frame position."""
        self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self.frame_pos / self.fps
    
    def _get_bottom_center(self, bbox: np.ndarray) -> Tuple[int, int]:
        """Ground-plane anchor point (feet position)."""
        x1, y1, x2, y2 = bbox.astype(int)
        return (x1 + x2) // 2, y2
    
    def _is_person_in_zone(self, results, frame_shape: Tuple[int, int]) -> Dict[int, float]:
        """Check persons in patrol zone & track dwell time."""
        zone_pts = self.zone_mapper.map_to_frame(frame_shape)
        current_time = self._get_current_video_time()
        active_ids = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf >= self.CONFIDENCE_THRESHOLD:
                        track_id = int(box.id[0]) if box.id is not None else -1
                        bbox = box.xyxy[0].cpu().numpy()
                        foot_point = self._get_bottom_center(bbox)
                        
                        if cv2.pointPolygonTest(zone_pts, foot_point, False) >= 0:
                            # Track dwell time for this person
                            if track_id not in self.person_dwell_times:
                                self.person_dwell_times[track_id] = current_time
                            dwell_time = current_time - self.person_dwell_times[track_id]
                            active_ids[track_id] = dwell_time
        
        return active_ids
    
    def _update_compliance_state(self, person_presence: bool, current_time: float):
        """Update patrol compliance state machine."""
        elapsed = current_time - self.last_patrol_timestamp
        
        # Check for patrol completion (2s dwell time)
        if person_presence and self.person_stop_timer >= self.PERSON_STOP_DURATION:
            self.last_patrol_timestamp = current_time
            self.person_dwell_times.clear()
            self.person_stop_timer = 0.0
            self.alert_latch = False
            self.current_state = 0  # PATROLLED
            logger.info(f"✅ PATROL VERIFIED @ {current_time:.1f}s (Trackers reset)")
            return
        
        # State transitions based on elapsed time
        if elapsed > self.PATROL_INTERVAL_SECONDS:
            self.current_state = 2  # BREACH
            if not self.alert_latch:
                overrun = elapsed - self.PATROL_INTERVAL_SECONDS
                logger.warning(f"🚨 [ALERT] Patrol Missed! Overrun: {overrun:.1f}s")
                self.alert_latch = True
        elif elapsed > (self.PATROL_INTERVAL_SECONDS * 0.5):
            self.current_state = 1  # PENDING
        else:
            self.current_state = 0  # Within tolerance
    
    def _draw_status_banner(self, frame: np.ndarray) -> np.ndarray:
        """Render comprehensive patrol status HUD."""
        h, w = frame.shape[:2]
        color = self.STATE_COLORS[self.current_state]
        current_time = self._get_current_video_time()
        elapsed = current_time - self.last_patrol_timestamp
        remaining = max(0, self.PATROL_INTERVAL_SECONDS - elapsed)
        
        # Banner background
        banner_h = 80
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner[:] = color
        
        # Primary status
        status = self.STATE_LABELS[self.current_state]
        cv2.putText(banner, status, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)
        
        # Timer display
        timer_text = f"Next Due: {remaining:.0f}s"
        timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
        cv2.putText(banner, timer_text, (w - timer_size[0] - 25, 45), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)
        
        # Additional metrics
        metrics = [
            f"Last Patrol: {elapsed:.0f}s ago",
            f"FPS: {self.fps:.1f} | Frame: {self.frame_pos}"
        ]
        for i, metric in enumerate(metrics):
            cv2.putText(banner, metric, (25, 70 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        frame[:banner_h] = banner
        return frame
    
    def _draw_zone_and_people(self, results, frame: np.ndarray):
        """Render patrol zone & highlight guards inside."""
        zone_pts = self.zone_mapper.map_to_frame(frame.shape)
        color = self.STATE_COLORS[self.current_state]
        
        # Zone polygon
        cv2.polylines(frame, [zone_pts], True, color, 4)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_pts], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Persons in zone
        current_time = self._get_current_video_time()
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf >= self.CONFIDENCE_THRESHOLD:
                        track_id = int(box.id[0]) if box.id is not None else -1
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox.astype(int)
                        foot = self._get_bottom_center(bbox)
                        
                        if cv2.pointPolygonTest(zone_pts, foot, False) >= 0:
                            # Highlight guard
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.circle(frame, foot, 12, color, -1)
                            
                            # Dwell time label
                            dwell = current_time - self.person_dwell_times.get(track_id, 0)
                            label = f"ID:{track_id} | {dwell:.1f}s"
                            cv2.putText(frame, label, (x1, y1-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def run(self):
        """Main patrol monitoring loop."""
        logger.info("🚔 GarudAI Module 3: Guard Patrol Monitor")
        logger.info(f"⏱️  Patrol interval: {self.PATROL_INTERVAL_SECONDS}s")
        logger.info(f"📍 Zone: {self.zone_file}")
        
        cv2.namedWindow("GarudAI - Patrol Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GarudAI - Patrol Monitor", self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                # Infinite looping (preserve timestamps across loops)
                if not ret:
                    logger.debug("🔄 Video loop reset")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                current_time = self._get_current_video_time()
                
                # YOLO tracking
                results = self.model.track(frame, persist=True, conf=self.CONFIDENCE_THRESHOLD,
                                         classes=[0], verbose=False)
                
                # Patrol zone analysis
                person_data = self._is_person_in_zone(results, frame.shape)
                person_present = bool(person_data)
                
                if person_present:
                    self.person_stop_timer = current_time - min(person_data.values())
                else:
                    self.person_stop_timer = 0.0
                
                # Update compliance state
                self._update_compliance_state(person_present, current_time)
                
                # Rendering pipeline
                display_frame = frame.copy()
                self._draw_zone_and_people(results, display_frame)
                display_frame = self._draw_status_banner(display_frame)
                display_frame = cv2.resize(display_frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
                
                cv2.imshow("GarudAI - Patrol Monitor", display_frame)
                
                # Responsive controls
                key = cv2.waitKey(self.WAIT_KEY_DELAY) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("⏹️  Patrol monitor stopped")
                    break
                    
        except KeyboardInterrupt:
            logger.info("⌨️  Interrupted")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Patrol monitor shutdown")

def parse_args():
    parser = argparse.ArgumentParser(description='GarudAI Module 3: Guard Patrol Verification')
    parser.add_argument('-v', '--video', required=True, help='Demonstration video file')
    parser.add_argument('--zone', default='patrol_zone.json', help='Patrol checkpoint zone')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    monitor = PatrolComplianceEngine(args.video, args.zone)
    monitor.run()