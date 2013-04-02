% CSC 321 - Assignment 4
% Zeeshan Qureshi (997108954)
% 2 Apr 2013

Part 1
------

**rbmfun** parameters:

  + Hidden Units = 160
  + Weight Cost = 0.005
  + Epochs = 100

**a4** parameters:

  + Weight Decay = 0.001
  + Hidden Units = 160
  + Iterations = 4000
  + Learning Rate = 0.05
  + Momentum = 0.5
  + Early Stopping = True
  + Mini Batch  = 10

Lowest Error Rate (w/ Pre-training):
  : 8.45%

Lowest Error Rate (w/o Pre-training):
  : 11.08%

I initially started with $256$ hidden units and was able to reach about
$23$% error rate but the test loss was very wobbly at the end showing
that the learning rate was too high and it was stuck around the minima.
I then turned down the learning rate and increased the momentum value
which smoothed out the loss curve and also provided a very fast drop in
loss over the first few iterations. This gave me an error rate of
$8.8$%. I then tried out a different number of hidden units and reached
$8.45$% error at $160$ units.
