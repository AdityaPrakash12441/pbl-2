"""
Animal Behavior Analyzer
Detects and classifies animal behaviors based on motion, posture, and contextual patterns
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple

class BehaviorAnalyzer:
    """Analyzes animal behavior patterns"""
    
    def __init__(self, history_frames: int = 15):
        """
        Initialize behavior analyzer
        
        Args:
            history_frames: Number of frames to track for motion analysis
        """
        self.history_frames = history_frames
        self.object_trackers: Dict[str, dict] = {}  # Track objects across frames
        self.motion_history: Dict[str, deque] = {}  # Store motion history
    
    def analyze_behavior(self, 
                        frame: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        species: str,
                        object_id: str) -> Dict[str, any]:
        """
        Analyze behavior of detected animal
        
        Args:
            frame: Current video frame
            bbox: Bounding box (x1, y1, x2, y2)
            species: Detected species name
            object_id: Unique object identifier for tracking
            
        Returns:
            Dictionary with behavior analysis including:
            - behavior: Primary behavior classification
            - confidence: Confidence score (0-1)
            - motion_speed: Estimated motion speed (0-1)
            - direction: Direction of movement
            - posture: Body posture analysis
        """
        x1, y1, x2, y2 = bbox
        
        # Get center point and area
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        
        # Initialize or update tracker
        if object_id not in self.object_trackers:
            self.object_trackers[object_id] = {
                'positions': deque(maxlen=self.history_frames),
                'areas': deque(maxlen=self.history_frames),
                'frame_count': 0
            }
            self.motion_history[object_id] = deque(maxlen=self.history_frames)
        
        tracker = self.object_trackers[object_id]
        tracker['positions'].append((center_x, center_y))
        tracker['areas'].append(area)
        tracker['frame_count'] += 1
        
        # Extract ROI for posture analysis
        roi = frame[max(0, y1):min(frame.shape[0], y2), 
                   max(0, x1):min(frame.shape[1], x2)]
        
        # Calculate motion metrics
        motion_speed = self._calculate_motion_speed(tracker['positions'])
        motion_direction = self._calculate_direction(tracker['positions'])
        
        # Analyze posture from ROI
        posture_score = self._analyze_posture(roi) if roi.size > 0 else 0.5
        
        # Classify behavior based on metrics
        behavior, behavior_conf = self._classify_behavior(
            species,
            motion_speed,
            motion_direction,
            posture_score,
            tracker['areas']
        )
        
        # Store motion history for temporal analysis
        self.motion_history[object_id].append({
            'speed': motion_speed,
            'behavior': behavior,
            'timestamp': tracker['frame_count']
        })
        
        return {
            'behavior': behavior,
            'confidence': behavior_conf,
            'motion_speed': motion_speed,
            'direction': motion_direction,
            'posture': posture_score,
            'tracking_frames': tracker['frame_count']
        }
    
    def _calculate_motion_speed(self, positions: deque) -> float:
        """
        Calculate motion speed based on position history
        
        Returns:
            Normalized speed (0-1)
        """
        if len(positions) < 3:
            return 0.0
        
        positions_list = list(positions)
        distances = []
        
        for i in range(1, len(positions_list)):
            dx = positions_list[i][0] - positions_list[i-1][0]
            dy = positions_list[i][1] - positions_list[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        # Average motion distance, normalized to frame size (assume max movement is 100 pixels)
        avg_motion = np.mean(distances) if distances else 0
        normalized_speed = min(avg_motion / 100.0, 1.0)
        
        return normalized_speed
    
    def _calculate_direction(self, positions: deque) -> str:
        """
        Determine direction of movement
        
        Returns:
            Direction string: "Stationary", "Moving Up", "Moving Down", "Moving Left", "Moving Right", "Diagonal"
        """
        if len(positions) < 2:
            return "Stationary"
        
        positions_list = list(positions)
        recent_positions = positions_list[-5:] if len(positions_list) >= 5 else positions_list
        
        start = recent_positions[0]
        end = recent_positions[-1]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # If minimal movement, consider stationary
        if abs(dx) < 5 and abs(dy) < 5:
            return "Stationary"
        
        # Determine primary direction
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy * 2:
            return "Moving Left" if dx < 0 else "Moving Right"
        elif abs_dy > abs_dx * 2:
            return "Moving Up" if dy < 0 else "Moving Down"
        else:
            return "Diagonal"
    
    def _analyze_posture(self, roi: np.ndarray) -> float:
        """
        Analyze body posture from ROI
        
        Returns:
            Posture score (0-1): 0 = lying down, 1 = standing upright
        """
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return 0.5
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Analyze aspect ratio of contours
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w == 0:
            return 0.5
        
        # Aspect ratio: high ratio = upright, low ratio = lying down
        aspect_ratio = h / w
        
        # Normalize to 0-1 (aspect ratio typically 0.5-3)
        posture_score = min(aspect_ratio / 2.0, 1.0)
        
        return posture_score
    
    def _classify_behavior(self,
                          species: str,
                          motion_speed: float,
                          direction: str,
                          posture_score: float,
                          areas: deque) -> Tuple[str, float]:
        """
        Classify animal behavior based on multiple features
        
        Returns:
            Tuple of (behavior_name, confidence)
        """
        # Stationary behaviors
        if motion_speed < 0.1:
            if posture_score > 0.7:
                return "Standing", 0.85
            elif posture_score < 0.3:
                return "Resting/Lying Down", 0.85
            else:
                return "Idle", 0.75
        
        # Slow movement
        elif motion_speed < 0.3:
            if direction == "Stationary":
                return "Browsing/Grazing", 0.75
            else:
                return "Walking", 0.80
        
        # Moderate movement
        elif motion_speed < 0.6:
            if "predator" in species.lower() or any(x in species.lower() for x in 
                ['lion', 'tiger', 'leopard', 'wolf', 'hyena', 'cheetah']):
                return "Hunting/Stalking", 0.70
            else:
                return "Trotting", 0.75
        
        # High speed movement
        else:
            if any(x in species.lower() for x in ['cheetah', 'gazelle', 'antelope', 'deer']):
                return "Running/Fleeing", 0.80
            else:
                return "Fast Movement", 0.75
        
        return "Unknown Behavior", 0.5
    
    def cleanup_old_objects(self, max_age: int = 300):
        """
        Remove tracking data for objects that haven't been updated recently
        
        Args:
            max_age: Maximum frames to keep tracking data
        """
        current_frame = max([self.object_trackers[k]['frame_count'] 
                           for k in self.object_trackers], default=0)
        
        objects_to_remove = [
            obj_id for obj_id, tracker in self.object_trackers.items()
            if current_frame - tracker['frame_count'] > max_age
        ]
        
        for obj_id in objects_to_remove:
            if obj_id in self.object_trackers:
                del self.object_trackers[obj_id]
            if obj_id in self.motion_history:
                del self.motion_history[obj_id]

class BehaviorVisualizer:
    """Visualize behavior analysis on video frames"""
    
    @staticmethod
    def draw_behavior_info(frame: np.ndarray,
                          behavior_data: Dict[str, any],
                          bbox: Tuple[int, int, int, int],
                          species: str) -> np.ndarray:
        """
        Draw behavior information on frame
        
        Args:
            frame: Video frame to draw on
            behavior_data: Behavior analysis results
            bbox: Bounding box (x1, y1, x2, y2)
            species: Species name
            
        Returns:
            Frame with behavior visualization
        """
        x1, y1, x2, y2 = bbox
        
        # Behavior text
        behavior = behavior_data['behavior']
        confidence = behavior_data['confidence']
        motion_speed = behavior_data['motion_speed']
        direction = behavior_data['direction']
        
        # Create behavior label with emoji-like indicators
        behavior_text = f"{behavior} ({confidence:.0%})"
        
        # Draw behavior box below species box
        behavior_size = cv2.getTextSize(behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        behavior_y = y2 + behavior_size[1] + 15
        
        # Color based on behavior confidence
        if confidence > 0.8:
            behavior_color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.6:
            behavior_color = (0, 255, 255)  # Yellow - medium confidence
        else:
            behavior_color = (0, 165, 255)  # Orange - low confidence
        
        # Draw behavior background
        cv2.rectangle(frame, (x1, y2 + 5),
                     (x1 + behavior_size[0] + 5, behavior_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(frame, behavior_text, (x1 + 3, behavior_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, behavior_color, 1)
        
        # Draw motion indicator arrow
        if motion_speed > 0.1:
            BehaviorVisualizer._draw_motion_indicator(frame, bbox, direction, motion_speed)
        
        # Draw motion speed bar
        BehaviorVisualizer._draw_speed_bar(frame, bbox, motion_speed)
        
        return frame
    
    @staticmethod
    def _draw_motion_indicator(frame: np.ndarray,
                              bbox: Tuple[int, int, int, int],
                              direction: str,
                              speed: float):
        """Draw motion direction and speed indicator"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        arrow_length = int(20 * speed)
        arrow_color = (0, 255, 255)  # Cyan
        
        # Direction to angle mapping
        direction_angles = {
            "Moving Up": (0, -arrow_length),
            "Moving Down": (0, arrow_length),
            "Moving Left": (-arrow_length, 0),
            "Moving Right": (arrow_length, 0),
            "Diagonal": (int(arrow_length * 0.7), int(arrow_length * 0.7))
        }
        
        if direction in direction_angles:
            dx, dy = direction_angles[direction]
            end_x = center_x + dx
            end_y = center_y + dy
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                           arrow_color, 2, tipLength=0.3)
    
    @staticmethod
    def _draw_speed_bar(frame: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       speed: float):
        """Draw motion speed visualization bar"""
        x1, y1, x2, y2 = bbox
        
        # Draw speed bar below bounding box
        bar_width = x2 - x1
        bar_height = 4
        bar_x = x1
        bar_y = y2 + 2
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Fill based on speed
        filled_width = int(bar_width * speed)
        speed_color = (0, 255, 0) if speed < 0.5 else (0, 165, 255) if speed < 0.8 else (0, 0, 255)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                     speed_color, -1)
        
        return frame
