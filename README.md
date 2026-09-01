# Opencv-measurement-objects
OpenCV-based object detection, measurement, contour and shape analysis.
# OpenCV Object Measurement & Image Processing

This is a computer vision project I built using Python and OpenCV.

The goal of the project is to detect objects in an image and automatically
calculate useful measurements for each object.

Instead of manually measuring every object, the program processes the image,
finds the objects, and generates the measurements automatically.

## What it does

The program can:

- Detect objects from an image
- Segment the image
- Remove small noise
- Find object contours
- Filter objects based on area
- Count detected objects
- Find the center (centroid) of each object
- Calculate perimeter
- Find bounding boxes
- Measure width and height
- Calculate circularity
- Calculate convex hull
- Calculate solidity
- Label detected objects
- Save the measurements to a CSV file

## How it works

The basic pipeline is:

Image  
↓  
Grayscale  
↓  
Segmentation  
↓  
Morphological processing  
↓  
Contour detection  
↓  
Area filtering  
↓  
Object measurements  
↓  
Annotation  
↓  
CSV + image outputs

## Measurements

For each detected object, the program calculates:

- Area
- Perimeter
- Centroid (X, Y)
- Bounding box
- Width
- Height
- Circularity
- Convex hull area
- Solidity

The measurements are currently in **pixels**.

For example:

```text
Area      → pixels²
Perimeter → pixels
Width     → pixels
Height    → pixels
Centroid  → pixel coordinates
