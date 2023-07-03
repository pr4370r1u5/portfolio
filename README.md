# portfolio
Included in this repository are examples of my coding from my graduate work at the Georgia Institute of Technology.

- L6-Implementing_SLAM_py3.py is the execution of a SLAM algorithm. This was part of a bigger lab teaching the concept.
- leak_detector.py is an algorithm that detects inconsistent water flow using data from a water meter and determines if there is a possible leak in the plumbing system. It uses previous use data from an individual to "train" the algorithm, then outputs a Boolean indicator based on analysis of moving averages against current use data.
- listrank_hj.c is a parallel list ranking algorithm, which takes an unordered linked list and sorts it according to the distance from the "head" of the list. It is implemented using OpenMP.
- seam_carving.py automatically expands a digital image along one-pixel "seams" of low-gradient data. It separates the image along each seam, then averages the data to create an optically seamless image. This maintains the high-gradient data, assumed to be the "subject" of the image. This was implemented using openCV.
- td_lambda.py is the implementation of a basic reinforcement learning algorithm. Temporal difference learning implements a "time window" over which relevant data is weighted.
- tridirectional astar.py is an implementation of the A* algorithm that searches for optimal paths between three origin points. Using a heap, the lowest-cost route connecting three nodes is found.
