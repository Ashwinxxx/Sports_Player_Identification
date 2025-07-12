from ultralytics import YOLO
model = YOLO('best.pt')
import cv2
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
import seaborn as sns

class PlayerDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, using default best.pt")
            model_path = 'best.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = YOLO(model_path).to(device)
    
    def detect(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        results = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.model(frame, conf=0.3, classes=[0])
            current_frame_detections = []

            if detections[0].boxes is not None:
                for box in detections[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    current_frame_detections.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,  #timestamp added 
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls
                    })
            results.append(current_frame_detections)
            frame_count += 1
        
        cap.release()
        return results
def save_detections_to_json(detections, output_filename):
    with open(output_filename, 'w') as f:
        json.dump(detections, f, indent=4)

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter_area = (xB - xA) * (yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)

def calculate_center_distance(boxA, boxB):
    cA = [(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2]
    cB = [(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2]
    return np.linalg.norm(np.array(cA) - np.array(cB))

def assign_consistent_ids(all_detections, threshold=0.3):
    player_id_counter = 0
    player_tracks = {}

    for frame_idx, frame_detections in enumerate(all_detections):
        updated = []
        for det in frame_detections:
            if det['class'] != 0:
                updated.append(det)
                continue

            bbox = det['bbox']
            best_score, best_id = -1, -1
            for pid, track in player_tracks.items():
                last_bbox = track['last_bbox']
                iou = calculate_iou(bbox, last_bbox)
                distance = calculate_center_distance(bbox, last_bbox)
                distance_score = max(0, 1 - distance / 200)
                score = 0.7 * iou + 0.3 * distance_score
                if score > best_score and score > threshold:
                    best_score = score
                    best_id = pid

            if best_id != -1:
                det['player_id'] = best_id
                player_tracks[best_id]['last_bbox'] = bbox
                player_tracks[best_id]['history'].append(bbox)
            else:
                det['player_id'] = player_id_counter
                player_tracks[player_id_counter] = {'last_bbox': bbox, 'history': [bbox]}
                player_id_counter += 1
            updated.append(det)
        all_detections[frame_idx] = updated
    return all_detections

def plot_enhanced_trajectories(all_detections, video_name="video"):
    """Enhanced trajectory plotting with better spatial and temporal visualization"""
    print(f"Generating enhanced trajectory plot for {video_name}...")
    
    # Extract player paths with timestamps
    player_data = {}
    for frame_idx, frame in enumerate(all_detections):
        for det in frame:
            if 'player_id' in det:
                pid = det['player_id']
                bbox = det['bbox']
                center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                timestamp = det.get('timestamp', frame_idx)
                
                if pid not in player_data:
                    player_data[pid] = {'positions': [], 'timestamps': [], 'frames': []}
                
                player_data[pid]['positions'].append(center)
                player_data[pid]['timestamps'].append(timestamp)
                player_data[pid]['frames'].append(frame_idx)

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f'Enhanced Player Analysis - {video_name}', fontsize=18, y=0.98)
    
    # 1. Spatial Trajectory with Time Color Coding
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                            c=timestamps, cmap='plasma', 
                            s=30, alpha=0.8, label=f'Player {pid}')
        ax1.plot(positions[:, 0], positions[:, 1], 
                alpha=0.5, color=colors[i], linewidth=2)
        ax1.scatter(positions[0, 0], positions[0, 1], 
                   marker='o', s=150, color='green', alpha=0.9, edgecolor='black', linewidth=2)
        ax1.scatter(positions[-1, 0], positions[-1, 1], 
                   marker='s', s=150, color='red', alpha=0.9, edgecolor='black', linewidth=2)
    
    ax1.invert_yaxis()
    ax1.set_title('Spatial Trajectories (Green=Start, Red=End)', fontsize=14, pad=20)
    ax1.set_xlabel('X Position (pixels)', fontsize=12)
    ax1.set_ylabel('Y Position (pixels)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    if len(player_data) > 0:
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('Time (seconds)', fontsize=10)
    
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        x_offset = i * 20  
        ax2.scatter(timestamps, positions[:, 0] + x_offset, alpha=0.8, s=25, 
                   color=colors[i], label=f'Player {pid}', edgecolor='black', linewidth=0.5)
        ax2.plot(timestamps, positions[:, 0] + x_offset, alpha=0.6, 
                color=colors[i], linewidth=2)
    
    ax2.set_title('X-Position vs Time', fontsize=14, pad=20)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('X Position (pixels)', fontsize=12)
    ax2.set_xlim(0, max([max(data['timestamps']) for data in player_data.values()]) if player_data else 10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3 = axes[0, 2]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        y_offset = i * 20 
        ax3.scatter(timestamps, positions[:, 1] + y_offset, alpha=0.8, s=25, 
                   color=colors[i], label=f'Player {pid}', edgecolor='black', linewidth=0.5)
        ax3.plot(timestamps, positions[:, 1] + y_offset, alpha=0.6, 
                color=colors[i], linewidth=2)
    
    ax3.set_title('Y-Position vs Time', fontsize=14, pad=20)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Y Position (pixels)', fontsize=12)
    ax3.set_xlim(0, max([max(data['timestamps']) for data in player_data.values()]) if player_data else 10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 4. Spatial Trajectory (Standard X-Y View)
    ax4 = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        scatter = ax4.scatter(positions[:, 0], positions[:, 1], 
                            c=timestamps, cmap='viridis', 
                            s=30, alpha=0.8, label=f'Player {pid}', 
                            edgecolor='black', linewidth=0.5)
        ax4.plot(positions[:, 0], positions[:, 1], 
                alpha=0.5, color=colors[i], linewidth=2)

        ax4.scatter(positions[0, 0], positions[0, 1], 
                   marker='o', s=150, color='green', alpha=0.9, edgecolor='black', linewidth=2)
        ax4.scatter(positions[-1, 0], positions[-1, 1], 
                   marker='s', s=150, color='red', alpha=0.9, edgecolor='black', linewidth=2)
    
    ax4.invert_yaxis()
    ax4.set_title('Spatial Trajectories (X-Y View)', fontsize=14, pad=20)
    ax4.set_xlabel('X Position (pixels)', fontsize=12)
    ax4.set_ylabel('Y Position (pixels)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    if len(player_data) > 0:
        cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar.set_label('Time (seconds)', fontsize=10)
    
    # 5. Movement Speed Analysis
    ax5 = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            time_diffs = np.diff(timestamps)
            time_diffs[time_diffs == 0] = 1  
            speeds = distances / time_diffs
            
            # Plot speed over time with markers
            ax5.plot(timestamps[1:], speeds, alpha=0.8, 
                    color=colors[i], label=f'Player {pid}', linewidth=2, marker='o', markersize=4)
    
    ax5.set_title('Player Movement Speed Over Time', fontsize=14, pad=20)
    ax5.set_xlabel('Time (seconds)', fontsize=12)
    ax5.set_ylabel('Speed (pixels/second)', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined X-Y Position vs Time
    ax6 = axes[1, 2]
    colors = plt.cm.tab10(np.linspace(0, 1, len(player_data)))
    
    for i, (pid, data) in enumerate(player_data.items()):
        positions = np.array(data['positions'])
        timestamps = np.array(data['timestamps'])
        
        # Plot both X and Y positions on same graph
        ax6.plot(timestamps, positions[:, 0], alpha=0.8, 
                color=colors[i], label=f'Player {pid} (X)', linewidth=2, linestyle='-')
        ax6.plot(timestamps, positions[:, 1], alpha=0.8, 
                color=colors[i], label=f'Player {pid} (Y)', linewidth=2, linestyle='--')
    
    ax6.set_title('Combined X & Y Position vs Time', fontsize=14, pad=20)
    ax6.set_xlabel('Time (seconds)', fontsize=12)
    ax6.set_ylabel('Position (pixels)', fontsize=12)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{video_name}_enhanced_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_spatial_temporal_stats(all_detections):
    """Generate detailed spatial and temporal statistics"""
    stats = {
        'total_players': 0,
        'total_frames': len(all_detections),
        'player_stats': {}
    }
    
    player_data = {}
    for frame_idx, frame in enumerate(all_detections):
        for det in frame:
            if 'player_id' in det:
                pid = det['player_id']
                bbox = det['bbox']
                center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                
                if pid not in player_data:
                    player_data[pid] = {
                        'positions': [],
                        'frames': [],
                        'first_seen': frame_idx,
                        'last_seen': frame_idx
                    }
                
                player_data[pid]['positions'].append(center)
                player_data[pid]['frames'].append(frame_idx)
                player_data[pid]['last_seen'] = frame_idx
    
    stats['total_players'] = len(player_data)
    
    for pid, data in player_data.items():
        positions = np.array(data['positions'])
        frames = np.array(data['frames'])
        total_distance = 0
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
        
        avg_position = np.mean(positions, axis=0)
        position_variance = np.var(positions, axis=0)
        
        stats['player_stats'][pid] = {
            'total_detections': len(positions),
            'total_distance_moved': float(total_distance),
            'avg_position': avg_position.tolist(),
            'position_variance': position_variance.tolist(),
            'first_seen_frame': int(data['first_seen']),
            'last_seen_frame': int(data['last_seen']),
            'frames_active': int(data['last_seen'] - data['first_seen'] + 1)
        }
    
    return stats

def main():
    model_path = 'best.pt'
    broadcast_video_path = 'broadcast.mp4'
    tacticam_video_path = 'tacticam.mp4'

    detector = PlayerDetector(model_path)

    print("\n--- Processing Broadcast Video ---")
    broadcast_detections = detector.detect(broadcast_video_path)
    save_detections_to_json(broadcast_detections, 'broadcast_detections.json')
    broadcast_with_ids = assign_consistent_ids(broadcast_detections)
    
    # Generate enhanced visualizations
    plot_enhanced_trajectories(broadcast_with_ids, "broadcast")
    
    # Generate statistics
    broadcast_stats = generate_spatial_temporal_stats(broadcast_with_ids)
    with open('broadcast_stats.json', 'w') as f:
        json.dump(broadcast_stats, f, indent=2)
    
    print(f"Broadcast Analysis:")
    print(f"  - Total players detected: {broadcast_stats['total_players']}")
    print(f"  - Total frames processed: {broadcast_stats['total_frames']}")
    
    print("\n--- Processing Tacticam Video ---")
    tacticam_detections = detector.detect(tacticam_video_path)
    save_detections_to_json(tacticam_detections, 'tacticam_detections.json')
    tacticam_with_ids = assign_consistent_ids(tacticam_detections)
    
    # Generate enhanced visualizations
    plot_enhanced_trajectories(tacticam_with_ids, "tacticam")
    
    # Generate statistics
    tacticam_stats = generate_spatial_temporal_stats(tacticam_with_ids)
    with open('tacticam_stats.json', 'w') as f:
        json.dump(tacticam_stats, f, indent=2)
    
    print(f"Tacticam Analysis:")
    print(f"  - Total players detected: {tacticam_stats['total_players']}")
    print(f"  - Total frames processed: {tacticam_stats['total_frames']}")

if __name__ == '__main__':
    main()