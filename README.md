# Real-Time Height Estimation with Crowd Analytics

## Overview

This project estimates a person's height in real time using a standard webcam, YOLOv8, and ByteTrack. Detected heights are stored in a SQLite database and visualized through a Streamlit dashboard that shows crowd statistics such as average height and height distribution.

## How to Run

### Install Dependencies

```bash
pip install ultralytics opencv-python streamlit pandas matplotlib
```

### Run Height Detection

```bash
python main.py
```

### Run Dashboard

```bash
streamlit run app.py
```

## Calibration Method

A standard A4 sheet is used as a reference object for calibration.

1. Fix the A4 sheet vertically on a wall.
2. Start the application.
3. Click the top and bottom of the A4 sheet.
4. The system measures the sheet's height in pixels and calculates a pixel-to-centimeter conversion factor.

Formula:

```text
cm_per_pixel = Real Height of A4 Sheet / Height of A4 Sheet in Pixels
```

The detected person's height in pixels is then converted to centimeters using this scale.

For best accuracy, the camera and A4 sheet should remain fixed, and the person should stand close to the reference object.
