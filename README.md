(Option 1)
# 🏃‍♂️ Sports Player Identification (Multi-Camera View)  
*Internship Assessment Submission — by Ashwin S.*
 
Overview 

This project performs **sports player detection and ID tracking** using **YOLOv8** across two synchronized video feeds:
- 📺 `broadcast.mp4` — the traditional game view
- 🎥 `tacticam.mp4` — a static tactical overhead view

The goal is to **detect players**, **track them frame-by-frame**, and attempt to assign **consistent IDs** across both camera views. Outputs include bounding boxes, trajectory visualizations, and frame-by-frame JSON logs.
Key Features

Custom-trained YOLOv8 model** for robust sports player detection
Dual-camera input**: handles `broadcast` and `tacticam` videos in sync
Player ID tracking** across frames using spatial heuristics
Trajectory and timeline visualizations** to track movement
JSON export** of all frame-level detections and IDs
   
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

The system generates braodacst JSON file containing:
Frame-by-frame detection results
Player movement paths and coordinates
Cross-view player correspondence
Temporal appearance data

Visualizations

Trajectory Plots: Visual representation of player movements
Timeline Charts: Frame-wise player appearance patterns
Mapping Diagrams: Cross-view player correspondence visualization
Setup

Clone the repository:

bashgit clone https://github.com/Ashwinxxx/Sports_Player_Identification.git
cd Sports_Player_Identification

Install required dependencies:

bashpip install -r dependancies.txt

