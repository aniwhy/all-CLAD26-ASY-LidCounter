All-CLAD 2026<br/>
Anirudh Yuvaraj, Jonathan Philip<br/>
Problem #1: Lid Line Detection<br/>
Solution: Create an AI-powered object detection tool to track number of lids<br/>
--------------------------------------------------------------------------------------------------------
Model Used: YOLO11s<br/>
Dataset: Custom<br/>
↘️ Image Count: 1,193<br/>
Accuracy: 99.1% mAP<br/>
Speed: 80ms–120ms latency (Real-time edge processing)<br/>
Target Classes: lid_handle, hand<br/>

Logic Used:<br/>
  Frame-by-frame stability buffer to ensure model doesn't use "ghost," or false detections<br/>
  Automatically detects stacks of lids and efficiently counts them as one batch<br/>
  Subtraction logic uses hand detection to alter the count to ensure lid grabbing doesn't throw off count<br/>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Project Link: https://asy-allclad-lidcounter.streamlit.app/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Controls:<br/>
[p] to pause program<br/>
[q] to quit program<br/>
[r] to reset COUNT<br/>
