import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict
import argparse
import os
from pathlib import Path

class CentroidTracker:
    """Enhanced centroid-based tracker with improved duplicate handling"""
    def __init__(self, max_disappeared=50, max_distance=120):
        self.next_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Enhanced tracking features
        self.object_sizes = {}  # Track bounding box sizes
        self.velocities = {}    # Track movement velocities
        self.frame_count = 0
        self.min_distance_between_objects = 40  # Minimum distance between different objects
        
        # Occlusion handling
        self.last_known_positions = {}  # Store last known positions for disappeared objects
        self.reappear_distance_threshold = 80  # Distance to consider reappearance of same object
        
    def register(self, centroid, bbox_size=None):
        # Check if this centroid is too close to existing objects
        for existing_id, existing_centroid in self.objects.items():
            if self.disappeared[existing_id] == 0:  # Only check active objects
                distance = np.linalg.norm(np.array(centroid) - np.array(existing_centroid))
                if distance < self.min_distance_between_objects:
                    return
        
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.velocities[self.next_id] = [0, 0]  # [vx, vy]
        if bbox_size:
            self.object_sizes[self.next_id] = bbox_size
        self.next_id += 1

    def deregister(self, object_id):
        # Store last known position before deregistering
        if object_id in self.objects:
            self.last_known_positions[object_id] = self.objects[object_id].copy()
        
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.velocities:
            del self.velocities[object_id]
        if object_id in self.object_sizes:
            del self.object_sizes[object_id]

    def predict_position(self, object_id):
        """Predict next position based on velocity"""
        if object_id in self.velocities:
            current_pos = self.objects[object_id]
            velocity = self.velocities[object_id]
            # Limit prediction distance to avoid wild predictions
            max_prediction = 50
            pred_x = current_pos[0] + np.clip(velocity[0], -max_prediction, max_prediction)
            pred_y = current_pos[1] + np.clip(velocity[1], -max_prediction, max_prediction)
            return [pred_x, pred_y]
        return self.objects[object_id]

    def update_velocity(self, object_id, old_pos, new_pos):
        """Update velocity with smoothing"""
        if object_id in self.velocities:
            new_velocity = [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]]
            # Smooth velocity with previous velocity (momentum)
            alpha = 0.6  # Reduced smoothing for more responsive tracking
            self.velocities[object_id] = [
                alpha * new_velocity[0] + (1 - alpha) * self.velocities[object_id][0],
                alpha * new_velocity[1] + (1 - alpha) * self.velocities[object_id][1]
            ]

    def check_for_reappearing_objects(self, input_centroids, bbox_sizes):
        """Check if any new detections match recently disappeared objects"""
        reappear_matches = {}
        
        for i, centroid in enumerate(input_centroids):
            best_match_id = None
            best_distance = float('inf')
            
            # Check against recently disappeared objects
            for disappeared_id, last_pos in self.last_known_positions.items():
                distance = np.linalg.norm(np.array(centroid) - np.array(last_pos))
                
                # Also consider size similarity if available
                size_similarity = 1.0
                if disappeared_id in self.object_sizes and i < len(bbox_sizes):
                    old_size = self.object_sizes[disappeared_id]
                    new_size = bbox_sizes[i]
                    size_diff = abs(old_size[0] - new_size[0]) + abs(old_size[1] - new_size[1])
                    size_similarity = max(0.1, 1.0 - (size_diff / 200))  # Normalize size difference
                
                adjusted_distance = distance / size_similarity
                
                if adjusted_distance < self.reappear_distance_threshold and adjusted_distance < best_distance:
                    best_distance = adjusted_distance
                    best_match_id = disappeared_id
            
            if best_match_id is not None:
                reappear_matches[i] = best_match_id
        
        return reappear_matches

    def reactivate_object(self, old_id, centroid, bbox_size):
        """Reactivate a previously disappeared object"""
        self.objects[old_id] = centroid
        self.disappeared[old_id] = 0
        if bbox_size:
            self.object_sizes[old_id] = bbox_size
        # Remove from last known positions since it's active again
        if old_id in self.last_known_positions:
            del self.last_known_positions[old_id]

    def filter_overlapping_detections(self, detections, overlap_threshold=0.4):
        """Remove overlapping detections using Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Convert to numpy array for easier processing
        boxes = np.array(detections)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence (last column) if available, otherwise by area
        if boxes.shape[1] > 4:
            indices = np.argsort(boxes[:, 4])[::-1]  # Sort by confidence descending
        else:
            indices = np.argsort(areas)[::-1]  # Sort by area descending
        
        keep = []
        while len(indices) > 0:
            # Keep the detection with highest confidence/area
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining detections
            remaining_indices = indices[1:]
            current_box = boxes[current]
            remaining_boxes = boxes[remaining_indices]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = areas[current]
            remaining_areas = areas[remaining_indices]
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-10)
            
            # Keep only detections with IoU below threshold
            indices = remaining_indices[iou < overlap_threshold]
        
        return [detections[i] for i in keep]

    def update(self, detections):
        self.frame_count += 1
        
        # Filter overlapping detections first
        detections = self.filter_overlapping_detections(detections, overlap_threshold=0.4)
        
        if len(detections) == 0:
            # No detections - increment disappeared counter for all objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        # Extract centroids and sizes from detections
        input_centroids = []
        bbox_sizes = []
        for box in detections:
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            w, h = box[2] - box[0], box[3] - box[1]
            input_centroids.append([cx, cy])
            bbox_sizes.append([w, h])
        
        input_centroids = np.array(input_centroids)

        # Check for reappearing objects first (before registering new ones)
        reappear_matches = self.check_for_reappearing_objects(input_centroids, bbox_sizes)

        if len(self.objects) == 0:
            # No existing objects - but check for reappearing ones first
            for i, centroid in enumerate(input_centroids):
                if i in reappear_matches:
                    # Reactivate old object
                    old_id = reappear_matches[i]
                    self.reactivate_object(old_id, centroid, bbox_sizes[i])
                else:
                    # Register as new object
                    self.register(centroid, bbox_sizes[i])
        else:
            # Use predicted positions for better matching
            predicted_centroids = []
            object_ids = list(self.objects.keys())
            
            for obj_id in object_ids:
                if self.disappeared[obj_id] <= 2:  # Only predict for recently active tracks
                    predicted_pos = self.predict_position(obj_id)
                else:
                    predicted_pos = self.objects[obj_id]
                predicted_centroids.append(predicted_pos)
            
            predicted_centroids = np.array(predicted_centroids)
            
            # Compute distance matrix with size and velocity consistency
            D = np.linalg.norm(predicted_centroids[:, np.newaxis] - input_centroids, axis=2)
            
            # Add size similarity bonus (if we have size info)
            for i, obj_id in enumerate(object_ids):
                if obj_id in self.object_sizes:
                    for j, new_size in enumerate(bbox_sizes):
                        old_size = self.object_sizes[obj_id]
                        size_diff = abs(old_size[0] - new_size[0]) + abs(old_size[1] - new_size[1])
                        size_penalty = min(size_diff * 0.05, 30)  # Reduced size penalty
                        D[i, j] += size_penalty
            
            # Hungarian algorithm-like assignment with stricter distance threshold
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Assign existing objects to closest detections
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                distance = D[row, col]
                if distance > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                new_centroid = input_centroids[col]
                
                # Update position and velocity
                self.objects[object_id] = new_centroid
                self.update_velocity(object_id, old_centroid, new_centroid)
                self.disappeared[object_id] = 0
                
                # Update size with more conservative smoothing
                if object_id in self.object_sizes:
                    old_size = self.object_sizes[object_id]
                    new_size = bbox_sizes[col]
                    alpha = 0.2  # More conservative size updates
                    self.object_sizes[object_id] = [
                        alpha * new_size[0] + (1 - alpha) * old_size[0],
                        alpha * new_size[1] + (1 - alpha) * old_size[1]
                    ]
                else:
                    self.object_sizes[object_id] = bbox_sizes[col]
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                # Only predict position for very recently disappeared objects
                if self.disappeared[object_id] <= 2:
                    predicted_pos = self.predict_position(object_id)
                    self.objects[object_id] = predicted_pos
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Handle unmatched detections - check for reappearing objects first
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # Filter out detections that match reappearing objects
            remaining_unused_cols = []
            for col in unused_col_indices:
                if col in reappear_matches:
                    # Reactivate old object
                    old_id = reappear_matches[col]
                    self.reactivate_object(old_id, input_centroids[col], bbox_sizes[col])
                else:
                    remaining_unused_cols.append(col)
            
            # Register truly new objects
            for col in remaining_unused_cols:
                self.register(input_centroids[col], bbox_sizes[col])
        
        return self.objects.copy()

class PlayerTracker:
    """Main player tracking class with YOLOv11 + Centroid tracking"""
    
    def __init__(self, model_path, player_class_id=2, conf_threshold=0.5):
        """
        Initialize the player tracker
        
        Args:
            model_path (str): Path to YOLOv11 .pt model file
            player_class_id (int): Class ID for players in your model (default: 2)
            conf_threshold (float): Confidence threshold for detections
        """
        self.model_path = model_path
        self.player_class_id = player_class_id
        self.conf_threshold = conf_threshold
        
        # Load YOLO model
        print(f"Loading YOLOv11 model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize tracker
        print("Using Enhanced Centroid tracker...")
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=120)
        
        # Tracking history for visualization
        self.track_history = defaultdict(list)
        self.colors = {}
        
    def generate_color(self, track_id):
        """Generate consistent color for each track ID"""
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def detect_players(self, frame):
        """Detect players in frame using YOLO"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        player_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == self.player_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        player_detections.append([x1, y1, x2, y2, conf])
        
        return player_detections
    
    def update_tracks(self, detections, frame):
        """Update tracks with new detections"""
        # Centroid tracker expects [x1, y1, x2, y2] format
        detection_boxes = [[det[0], det[1], det[2], det[3]] for det in detections]
        tracked_objects = self.tracker.update(detection_boxes)
        
        # Convert centroids back to bounding boxes using tracked sizes
        box_tracked_objects = {}
        for track_id, centroid in tracked_objects.items():
            if track_id in self.tracker.object_sizes:
                # Use tracked size information
                w, h = self.tracker.object_sizes[track_id]
                x1 = centroid[0] - w/2
                y1 = centroid[1] - h/2
                x2 = centroid[0] + w/2
                y2 = centroid[1] + h/2
            elif detection_boxes:
                # Fallback to closest detection size
                min_dist = float('inf')
                closest_box = detection_boxes[0]
                for box in detection_boxes:
                    box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    dist = np.linalg.norm(np.array(centroid) - np.array(box_center))
                    if dist < min_dist:
                        min_dist = dist
                        closest_box = box
                
                w = closest_box[2] - closest_box[0]
                h = closest_box[3] - closest_box[1]
                x1 = centroid[0] - w/2
                y1 = centroid[1] - h/2
                x2 = centroid[0] + w/2
                y2 = centroid[1] + h/2
            else:
                # Default size fallback
                w, h = 60, 120  # Typical player size
                x1 = centroid[0] - w/2
                y1 = centroid[1] - h/2
                x2 = centroid[0] + w/2
                y2 = centroid[1] + h/2
            
            box_tracked_objects[track_id] = [max(0, x1), max(0, y1), x2, y2]
        
        return box_tracked_objects
    
    def draw_tracks(self, frame, tracked_objects):
        """Draw tracking results on frame"""
        for track_id, bbox in tracked_objects.items():
            x1, y1, x2, y2 = map(int, bbox)
            color = self.generate_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"Player {track_id}"
            font_scale = 0.6
            thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 6, y1), color, -1)
            
            # White text
            cv2.putText(frame, label, (x1 + 3, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Update track history
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self.track_history[track_id].append(center)
            
            # Keep only last 15 points for memory management
            if len(self.track_history[track_id]) > 15:
                self.track_history[track_id] = self.track_history[track_id][-15:]
            
            # Draw current position indicator
            cv2.circle(frame, center, 3, color, -1)
            cv2.circle(frame, center, 3, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self, input_video_path, output_video_path=None, show_live=False):
        """Process video with player tracking"""
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path specified
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_video_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect players
                detections = self.detect_players(frame)
                
                # Update tracks
                tracked_objects = self.update_tracks(detections, frame)
                
                # Draw tracking results
                frame = self.draw_tracks(frame, tracked_objects)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Players: {len(tracked_objects)}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing... {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Write frame if output specified
                if out:
                    out.write(frame)
                
                # Show live preview if requested
                if show_live:
                    cv2.imshow('Player Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Stopped by user")
                        break
                        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            if out:
                out.release()
            if show_live:
                cv2.destroyAllWindows()
            
            print("Processing completed!")
            
            # Print tracking statistics
            unique_players = len(self.track_history)
            print(f"Tracking Stats:")
            print(f"   Total unique players detected: {unique_players}")
            print(f"   Frames processed: {frame_count}")
            if unique_players > 0:
                avg_track_length = np.mean([len(track) for track in self.track_history.values()])
                print(f"   Average track length: {avg_track_length:.1f} frames")

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Player Tracker')
    parser.add_argument('--model', required=True, help='Path to YOLOv11 .pt model file')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', help='Path to output video file (optional)')
    parser.add_argument('--show', action='store_true', help='Show live tracking preview')
    parser.add_argument('--player-class', type=int, default=2, help='Player class ID (default: 2)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return
    
    # Set default output path if not specified
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_tracked.mp4")
    
    # Initialize tracker
    tracker = PlayerTracker(
        model_path=args.model,
        player_class_id=args.player_class,
        conf_threshold=args.conf
    )
    
    # Process video
    print(f"Starting player tracking...")
    print(f"   Model: {args.model}")
    print(f"   Video: {args.video}")
    print(f"   Output: {args.output}")
    print(f"   Player Class ID: {args.player_class}")
    print(f"   Confidence: {args.conf}")
    
    tracker.process_video(
        input_video_path=args.video,
        output_video_path=args.output,
        show_live=args.show
    )

if __name__ == "__main__":
    main()