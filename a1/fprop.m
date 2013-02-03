function [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(input_batch, model)
% Forward propagates through a neural network.
% Inputs:
%   input_batch: The input data as a matrix of size D X M, where,
%     D: number of input dimensions (in this case, number of words)
%     M: batchsize
%   model: Struct containing weights and biases of the network.
% Outputs:
%   embedding_layer_state: State of units in the embedding layer as a matrix of
%     size (numhid1 * D) X M
%
%   hidden_layer_state: State of units in the hidden layer as a matrix of size
%     numhid2 X M
%
%   output_layer_state: State of units in the output layer as a matrix of size
%     K X M
%

[numwords, batchsize] = size(input_batch);
[vocab_size, numhid1] = size(model.word_embedding_weights);
numhid2 = size(model.embed_to_hid_weights, 2);

%% COMPUTE STATE OF WORD EMBEDDING LAYER.
% Look up the inputs word indices in the word_embedding_weights matrix.
embedding_layer_state = reshape(...
  model.word_embedding_weights(reshape(input_batch, 1, []),:)',...
  numhid1 * numwords, []);

%% COMPUTE STATE OF HIDDEN LAYER.
% Compute inputs to hidden units.
inputs_to_hid = model.embed_to_hid_weights' * embedding_layer_state + ...
  repmat(model.hid_bias, 1, batchsize);

% Apply logistic activation function.
hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hid));

%% COMPUTE STATE OF OUTPUT LAYER.
inputs_to_softmax = model.hid_to_output_weights' * hidden_layer_state + ...
  repmat(model.output_bias, 1, batchsize);

% Subtract maximum. 
% Remember that adding or subtracting the same constant from each input to a
% softmax unit does not affect the outputs. So subtract the maximum to
% make all inputs <= 0. This prevents overflows when computing their
% exponents.
inputs_to_softmax = inputs_to_softmax...
  - repmat(max(inputs_to_softmax), vocab_size, 1);

% Compute exp.
output_layer_state = exp(inputs_to_softmax);

% Normalize to get probability distribution.
output_layer_state = output_layer_state ./ repmat(...
  sum(output_layer_state, 1), vocab_size, 1);
