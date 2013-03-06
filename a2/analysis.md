% CSC321: Assignment 2
% Zeeshan Qureshi (997108954)
% 6 Mar 2013

Programming
===========

Initially I did not want to look at the post on the forum and derive the
gradients from scratch but the relatively large amount of variables
compared to the example done in class caused a lot of confusion. After
looking at the example on the forum, the derivation wasn't particularly
hard but deciding on a consistent naming scheme was the key. After using
greek alphabets for weight matrices and capital letters for Input/Output
ones and writing down the indices used for each one of them it became
pretty clear what was going on.

These are the steps I followed in the gradient derivation:

  + Figure out the formula for the error E from the loss function
  + Figure out the chain of derivatives leading from E to the
    weights from hidden units -> classes
  + Similarly figure out the chain of derivatives from E to the
    weights from input -> hidden units

Once I had the formulas down for each unit, I vectorized the formula as in
the forum post for better performance.

Gradient Code
-------------

    ret.input_to_hid = (((model.hid_to_class' * (class_prob - data.targets))
        .* hid_output .* (1 - hid_output)) * data.inputs')
        / size(data.targets, 2);

    ret.hid_to_class = ((class_prob - data.targets) * hid_output')
        / size(data.targets, 2);

Result
------

After implementing the code and running *a2(0, 10, 30, 0.01, 0, false, 10)*
the training data classification loss was **2.301907**.
