import cv2
import numpy as np
import argparse
import json
import os
from typing import List, Tuple, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FenceCalibrator:
    """Interactive ROI/Polygon calibration utility for GarudAI."""
    
    DISPLAY_WIDTH = 960
    DISPLAY_HEIGHT = 540
    MIN_POINTS = 3
    POINT_RADIUS = 8
    LINE_THICKNESS = 3
    
    def __init__(self, source: str):
        self.source = source
        self.cap = None
        self.display_frame = None
        self.calib_frame = None
        self.points: List[Tuple[int, int]] = []
        self.norm_points: List[List[float]] = []
        self.frame_shape = None
        self.window_name = "GarudAI Fence Calibration"
        
        self._initialize_video()
        self._setup_window()
        self._setup_mouse_callback()
    
    def _initialize_video(self):
        """Initialize video capture and freeze first frame."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        
        self.calib_frame = frame.copy()
        self.frame_shape = frame.shape
        logger.info(f"✅ Frame captured: {self.frame_shape[1]}x{self.frame_shape[0]}")
    
    def _setup_window(self):
        """Create resizable calibration window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
        self.display_frame = cv2.resize(self.calib_frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
    
    def _setup_mouse_callback(self):
        """Install interactive mouse callback."""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x: int, y: int, flags, param):
        """Handle mouse events for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale mouse coordinates back to original frame resolution
            scale_x = self.frame_shape[1] / self.DISPLAY_WIDTH
            scale_y = self.frame_shape[0] / self.DISPLAY_HEIGHT
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            
            self.points.append((orig_x, orig_y))
            self._redraw()
            logger.info(f"📍 Point {len(self.points)}: ({orig_x:.1f}, {orig_y:.1f})")
    
    def _redraw(self):
        """Redraw frame with current points and polygon."""
        # Start with clean frame
        self.display_frame = cv2.resize(self.calib_frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        
        orig_points = np.array(self.points, np.int32)
        if len(self.points) >= 2:
            # Draw connecting lines
            for i in range(len(self.points) - 1):
                pt1 = (int(self.points[i][0] * self.DISPLAY_WIDTH / self.frame_shape[1]),
                       int(self.points[i][1] * self.DISPLAY_HEIGHT / self.frame_shape[0]))
                pt2 = (int(self.points[i+1][0] * self.DISPLAY_WIDTH / self.frame_shape[1]),
                       int(self.points[i+1][1] * self.DISPLAY_HEIGHT / self.frame_shape[0]))
                cv2.line(self.display_frame, pt1, pt2, (0, 255, 255), self.LINE_THICKNESS)
        
        if len(self.points) >= self.MIN_POINTS:
            # Draw closed polygon (yellow)
            display_pts = np.array([
                [int(p[0] * self.DISPLAY_WIDTH / self.frame_shape[1]),
                 int(p[1] * self.DISPLAY_HEIGHT / self.frame_shape[0])] 
                for p in self.points
            ], np.int32)
            cv2.polylines(self.display_frame, [display_pts], True, (0, 255, 255), self.LINE_THICKNESS)
            # Semi-transparent fill
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [display_pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
        
        # Draw all points (green circles)
        for i, (px, py) in enumerate(self.points):
            dx = int(px * self.DISPLAY_WIDTH / self.frame_shape[1])
            dy = int(py * self.DISPLAY_HEIGHT / self.frame_shape[0])
            cv2.circle(self.display_frame, (dx, dy), self.POINT_RADIUS, (0, 255, 0), -1)
            # Point number label
            cv2.putText(self.display_frame, str(i+1), (dx+10, dy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self._draw_instructions()
    
    def _draw_instructions(self):
        """Draw on-screen controls."""
        instructions = [
            "INSTRUCTIONS:",
            "• LEFT CLICK: Add polygon vertex",
            "• 'R': Reset/Clear points", 
            "• 'S': Save zone.json → EXIT",
            "• 'Q' or ESC: Quit (no save)",
            f"Points: {len(self.points)}/{self.MIN_POINTS}+ needed"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(self.display_frame, text, (15, y_pos),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    def _normalize_points(self) -> List[List[float]]:
        """Convert pixel coordinates to normalized (0.0-1.0)."""
        h, w = self.frame_shape[:2]
        return [[float(x)/w, float(y)/h] for x, y in self.points]
    
    def _save_zone_json(self):
        """Save normalized coordinates to zone.json."""
        if len(self.points) < self.MIN_POINTS:
            logger.warning(f"❌ Need ≥{self.MIN_POINTS} points! Got {len(self.points)}")
            return False
        
        self.norm_points = self._normalize_points()
        zone_data = {
            "name": "User-defined Surveillance Zone",
            "points": self.norm_points,  # Primary key
            "polygon": self.norm_points, # Backward compatibility
            "pixel_points": self.points, # Debug info
            "frame_shape": list(self.frame_shape),
            "calibrated_at": datetime.now().isoformat()
        }
        
        with open('zone.json', 'w') as f:
            json.dump(zone_data, f, indent=2)
        
        logger.info(f"✅ SAVED zone.json → {len(self.norm_points)} normalized points")
        print("\n" + "="*60)
        print("🎉 ZONE CALIBRATION COMPLETE!")
        print(f"📁 Saved to: zone.json")
        print("📊 Normalized points:")
        for i, pt in enumerate(self.norm_points):
            print(f"  Point {i+1}: [{pt[0]:.3f}, {pt[1]:.3f}]")
        print("="*60 + "\n")
        return True
    
    def run(self):
        """Main calibration loop."""
        logger.info("🎯 GarudAI Fence Calibration Started")
        logger.info(f"📹 Source: {self.source}")
        logger.info("👆 Click to place points, 'S' to save!")
        
        while True:
            cv2.imshow(self.window_name, self.display_frame)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r') or key == ord('R'):
                self.points.clear()
                logger.info("🔄 Points reset")
            
            elif key == ord('s') or key == ord('S'):
                if self._save_zone_json():
                    break
            
            elif key == ord('q') or key == 27:  # ESC
                logger.info("👋 Calibration cancelled")
                break
        
        self._cleanup()
    
    def _cleanup(self):
        """Clean shutdown."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Calibration utility closed")

def parse_args():
    parser = argparse.ArgumentParser(description='GarudAI Fence Calibration Utility')
    parser.add_argument('-v', '--video', type=str, 
                       help='Video file path (MP4)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine source
    if args.video:
        source = args.video
        logger.info(f"📁 Video mode: {source}")
    else:
        source = args.camera
        logger.info(f"📷 Camera mode: index {source}")
    
    try:
        calibrator = FenceCalibrator(source)
        calibrator.run()
    except KeyboardInterrupt:
        logger.info("⏹️  Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Error: {e}")

if __name__ == "__main__":
    main()