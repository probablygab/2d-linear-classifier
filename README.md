# 2D Linear Classifier

by Gabriel and Francisco - developed as an assignment for the Algorithms II course at UFMG

### Brief

This classifier is capable of creating an approximate linear model to separate classes in a binary fashion, a point *belongs* or *not* to a class.

A dataset is projected to 2D via PCA and every class is checked for a viable separation, if one is found a model is created and then evaluated. This is done in several steps:
  - Calculate convex hulls for every set of points
  - Perform a linear sweeping checking for hull intersections or hulls within hulls, this dictates whether a dataset is separable or not
  - If separable, calculate the perpendicular line equation which intersects the middle point of the shortest segment between hulls
  - A database can be "made" separable by manipulating points, if manipulated, a point is added to the test database
  - Given the line equation, run tests and evaluate the Precision, Recall and F1-Score of the model

### Algorithms

As part of the assignment, several algorithms are implemented from scratch:
  - Simple line geometry (orientation, turn direction, intersections)
  - Graham scan for convex hulls (does not use any trig functions)
  - Linear sweeping for intersection tests
  - Several other helper algorithms and functions

### Results

10 datasets were used to test the model, detailed discussion can be found in the "relatorio_tp1.ipynb" file

Some results can be seen below:

**Dataset 3: Dermatology**

*Classes view*

![hulls3](https://github.com/probablygab/2d-linear-classifier/assets/96994614/f4c6ab62-b886-4fd9-b356-4e504ad74597)

*Resulting model*

![output3](https://github.com/probablygab/2d-linear-classifier/assets/96994614/7a6d908a-fbbe-48a1-bf4c-d874ea679788)

**Dataset 1: Iris**

*Classes view*

![hull1](https://github.com/probablygab/2d-linear-classifier/assets/96994614/938ee987-feae-47d2-946f-d4f952dc4d21)

*Resulting model*

![output1](https://github.com/probablygab/2d-linear-classifier/assets/96994614/4dcee123-d580-44cb-b9f2-2edcb40e8a0b)

**Dataset 10: E. coli**

*Classes view*

![hulls10](https://github.com/probablygab/2d-linear-classifier/assets/96994614/54f064e1-9057-4ee2-a7fd-f9104fe4e938)

*Resulting model (unable to generate)*

![output10](https://github.com/probablygab/2d-linear-classifier/assets/96994614/dc712a6c-f473-4a57-9cda-c227b5cb6b22)
