This project uses YOLOv11 custom pretrained model  and OpenCV to detect players from two synchronized videos (broadcast and tacticam), assign consistent player IDs over time, and map players between views. It also provides visualizations of player trajectories and timelines.
This project demonstrates the power of computer vision in sports analytics, providing a foundation for advanced player identification and tracking systems.
Part of an assessment evaluation
Features

Player Detection using YOLOv11

Tracking: Assigns consistent IDs to players across frames

Player Mapping: Matches players between different camera views

Visual-Spatial Mapping: Trajectory plots for player movements

Temporal Mapping: Frame-wise appearance timelines
Output Formats
JSON Exports
The system generates braodacst JSON files containing:
Frame-by-frame detection results
Player movement paths and coordinates
Cross-view player correspondence
Temporal appearance data

Visualizations

Trajectory Plots: Visual representation of player movements
Timeline Charts: Frame-wise player appearance patterns
Mapping Diagrams: Cross-view player correspondence visualization

