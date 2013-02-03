function [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data()
% This method loads the training, validation and test sets.
% Outputs:
%   train_input: A matrix of size D X N_train.
%   train_target: A matrix of size 1 X N_train.
%   valid_input: A matrix of size D X N_valid.
%   valid_target: A matrix of size 1 X N_valid.
%   test_input: A matrix of size D X N_test.
%   test_target: A matrix of size 1 X N_test.
%   vocab: The vocabulary as a matrix of size 1 X K
%     Where -
%         D: number of input dimensions (in this case, 3).
%         N_train: number of training cases.
%         N_valid: number of validation cases.
%         N_test: number of test cases.
%         K: number of works in the vocabulary.
%     i.e., train_input(i, j) = k means that the ith word in training case j is
%     the word at index k of the vocabulary.

load data.mat;
numdims = size(data.trainData, 1);
D = numdims - 1;
train_input = data.trainData(1:D, :);
train_target = data.trainData(D + 1, :);
valid_input = data.validData(1:D, :);
valid_target = data.validData(D + 1, :);
test_input = data.testData(1:D, :);
test_target = data.testData(D + 1, :);
vocab = data.vocab;
end
