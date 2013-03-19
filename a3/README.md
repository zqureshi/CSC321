% CSC 321 - Assignment 3
% Zeeshan Qureshi (997108954)
% 19 Mar 2013

Part 1 - Fitting Gaussians
==========================

As can be seen in the figure, the gaussians had the highest validation
and training probability when *Gaussians = 5*. As the validation prob
rises till *5* and then starts to go down but the training probability
stays up means that the model is overfitting.

Be decreasing the standard deviation of the data, with lower gaussians
they do not tend to move at all. Thus the standard deviation is
affecting the maximum step size on each iteration.

![Fitting Number of Gaussians](plot-part1.png)\

\newpage

 Gaussians    Validation Prob    Training Prob
-----------  -----------------  ---------------
    1           7.9600             9.9341
    2           -26.8657           -10.5470
    3           -14.4738           8.3459
    4           13.5253            36.02819
    5           68.7349            100.6420
    6           63.1940            98.7709
    7           51.6067            98.9465
    8           58.3877            102.3457

