function [avg_CE] = evaluate(inputs, targets, model)
% Computes cross entropy.
% Inputs:
%   inputs: Matrix of shape D X N.
%   targets: Matrix of shape 1 X N.
%   model: Struct containing weights and biases of the network.
% Output:
%   Average Cross Entropy.
  input_size = size(inputs, 2);
  batchsize = 100;
  numbatches = floor(input_size / batchsize);
  K = size(model.vocab, 2);
  expansion_matrix = eye(K);
  count = 0;
  tiny = exp(-30);
  avg_CE = 0.0;
  for m = 1:numbatches
    input_batch = inputs(:, 1+(m-1)*batchsize : m*batchsize);
    target_batch = targets(:, 1+(m-1)*batchsize : m*batchsize);
    [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
      fprop(input_batch, model);
    expanded_target_batch = expansion_matrix(:, target_batch);
    CE = -sum(sum(...
      expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
    count =  count + 1;
    avg_CE = avg_CE + (CE - avg_CE) / count;
  end
end
