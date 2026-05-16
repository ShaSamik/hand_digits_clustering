========================================
Handwritten Digit Clustering Project
========================================

Files Included
----------------------------------------

1. kmean.py
   Main Python implementation of the K-Means clustering algorithm.

2. test.txt
   Input dataset containing handwritten digit images.
   Each row represents one image with 784 pixel values.

3. output.txt
   Output file containing the cluster assignment for each image.


========================================
Project Description
========================================

This project implements the K-Means clustering algorithm from scratch
to cluster handwritten digit images using unsupervised learning.

The dataset contains 10,740 grayscale digit images.
Each image is represented using 784 features corresponding to
a flattened 28x28 image.

The algorithm groups similar digit images together into K clusters
using Euclidean distance.


========================================
Dataset Format
========================================

Input File:
test.txt

Each line contains:
- 784 comma-separated pixel intensity values
- Pixel values range from 0 to 255

Example:

0,0,0,12,255,34,0,...

Each row represents one handwritten digit image.


========================================
How the Program Works
========================================

1. Load the dataset from test.txt
2. Randomly initialize K cluster centers
3. Assign each image to the nearest center
4. Update cluster centers using the mean of assigned points
5. Repeat until convergence or maximum iterations reached
6. Save final cluster assignments into output.txt


========================================
Features Used
========================================

- Euclidean distance
- Random initialization using existing data points
- Multiple runs with different random seeds
- SSE (Sum of Squared Errors) evaluation
- Convergence checking
- Optional normalization and preprocessing


========================================
Output Format
========================================

Output File:
output.txt

The file contains one cluster assignment per line.

Example:

3
3
7
1
9

Each number represents the cluster ID assigned to the
corresponding image in test.txt.


========================================
How to Run
========================================

Requirements:
- Python 3

Run the program using:

python3 kmean.py

or

python kmean.py


========================================
Notes
========================================

- The implementation was written mostly from scratch.
- Different random seeds may produce different clustering results.
- The best run is selected using the lowest SSE value.
- K-Means is sensitive to initialization, so multiple runs improve results.


========================================
Author
========================================

Shaharia Samik
