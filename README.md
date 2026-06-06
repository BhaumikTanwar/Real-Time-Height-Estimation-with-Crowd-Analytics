Overview
This project estimates a person's height in real time using a standard webcam and computer vision techniques. The system uses YOLOv8 for person detection and ByteTrack for tracking individuals across frames. 
Estimated heights are stored in a SQLite database and visualized through a Streamlit analytics dashboard.

Technologies Used
Python
OpenCV
YOLOv8 (Ultralytics)
ByteTrack
SQLite
Pandas
Streamlit
Matplotlib

Calibration Method

A standard A4 sheet is placed vertically on a wall and used as a reference object of known height.
Calibration Process -
Start the application.
Click the top of the A4 sheet.
Click the bottom of the A4 sheet.
The pixel height of the A4 sheet is measured.
A pixel-to-centimeter conversion factor is calculated.

Formula:
cm_per_pixel = A4_height_cm / A4_height_pixels

Running the Project
Install Dependencies
pip install ultralytics opencv-python streamlit pandas matplotlib
Start Height Detection
python main.py
Start Analytics Dashboard
streamlit run app.py
