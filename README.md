All-CLAD 2026
Anirudh Yuvaraj, Jonathan Philip
Problem #1: Lide Line Detection
Solution: Create an AI-powered object detection tool to track number of lids
--------------------------------------------------------------------------------------------------------
Model Used: YOLO11s
Dataset: Custom
↘️ Image Count: 1,193
Accuracy: 99.1% mAP)
Speed: 80ms–120ms latency (Real-time edge processing)
Target Classes: lid_handle, hand

Logic Used:
  15-frame stability buffer to ensure model doesn't use "ghost," or false detections
  Automatically detects stacks of lids and efficiently counts them as one batch
  Subtraction logic uses hand detection to alter the count to ensure lid grabbing doesn't throw off count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Dependencies (ensure Python library is installed):
  pip install -r requirements.txt
Frontend:
  streamlit run all_clad_app.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Controls:
[p] to pause program
[q] to quit program
[r] to reset COUNT
