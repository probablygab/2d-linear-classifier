# 2D Linear Classifier

by Gabriel and Francisco - developed as an assignment for the Algorithms II course at UFMG

### Brief

This classifier is capable of creating an approximate linear model to separate classes in a binary fashion, a point *belongs* or *not* to a class.

A dataset is projected to 2D via PCA and every class is checked for a viable separation, if one is found a model is created and then evaluated. This is done in several steps:
  - Calculate convex hulls for every set of points
  - Perform a linear sweeping checking for intersections or hulls within hulls, this dictates whether a dataset is separable (divisible) or not
  - If separable, calculate the perpendicular line equation which intersects the the middle point of the shortest segment between hulls
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

![hulls3](https://github.com/probablygab/tp1-alg2/assets/96994614/a619e371-2c31-42ca-b6b0-978b04936b36)

*Resulting model*

![output3](https://github.com/probablygab/tp1-alg2/assets/96994614/e639709d-a864-400b-9a01-aa9f89815231)

**Dataset 1: Iris**

*Classes view*

![hull1](https://github.com/probablygab/tp1-alg2/assets/96994614/1c390b9e-2bd1-4eeb-b219-4001325c3954)

*Resulting model*

![output1](https://github.com/probablygab/tp1-alg2/assets/96994614/36bdabfd-c502-447c-8361-974aea15c2d0)

**Dataset 10: E. coli**

*Classes view*

![hulls10](https://github.com/probablygab/tp1-alg2/assets/96994614/cb77aa30-079f-4e38-b743-281d0857d200)

*Resulting model (unable to generate)*

![output10](https://github.com/probablygab/tp1-alg2/assets/96994614/dba73268-4832-429c-a1fe-d27e83c3ff87)
