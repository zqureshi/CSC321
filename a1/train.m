% This function trains a neural network language model.
function [model] = train(d, num_hid)
% Inputs:
%   d: Number of dimensions in distributed word representation.
%   num_hid: Number of hidden units.
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.

start_time = clock;
% SET HYPERPARAMETERS HERE.
batchsize = 100;  % Mini-batch size.
learning_rate = 0.1;  % Learning rate; default = 0.1.
momentum = 0.9;  % Momentum; default = 0.9.
weight_cost = 0.0;  % Weight decay; default = 0.0.
epochs = 50;  % Maximum number of epochs.
numhid1 = d;  % Dimensionality of embedding space; default = 8.
numhid2 = num_hid;  % Number of units in hidden layer; default = 64.
init_wt = 0.01;  % Standard deviation of the normal distribution
                 % which is sampled to get the initial weights; default = 0.01

% VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100;
show_validation_CE_after = 1000;

% LOAD DATA.
[train_input, train_target, valid_input, valid_target, ...
  test_input, test_target, vocab] = load_data();
[D, train_set_size] = size(train_input);
numbatches = floor(train_set_size / batchsize);
K = size(vocab, 2);

% INITIALIZE WEIGHTS AND BIASES.
%   K: size of the vocabulary
%   D : number of words in the input.
%   numhid1: dimensionality of the embedding space.
%   numhid2: number of hidden units in the hidden layer.

% word_embedding_weights: Weights between input layer and word embedding layer.
model.word_embedding_weights = init_wt * randn(K, numhid1);

% embed_to_hid_weights: Weights between word embedding layer and hidden layer.
model.embed_to_hid_weights = init_wt * randn(D * numhid1, numhid2);

% hid_to_output_weights: Weights between hidden layer and output softmax unit.
model.hid_to_output_weights = init_wt * randn(numhid2, K);

% hid_bias: Bias of the hidden layer.
model.hid_bias = zeros(numhid2, 1);

% output_bias: Bias of the output layer.
model.output_bias = zeros(K, 1);

% vocab: The vocabulary.
model.vocab = vocab;

word_embedding_weights_delta = zeros(K, numhid1);
word_embedding_weights_grad = zeros(K, numhid1);
embed_to_hid_weights_delta = zeros(D * numhid1, numhid2);
hid_to_output_weights_delta = zeros(numhid2, K);
hid_bias_delta = zeros(numhid2, 1);
output_bias_delta = zeros(K, 1);
expansion_matrix = eye(K);
tiny = exp(-30);

% TRAIN.
best_valid_CE = Inf;
end_training = false;
for epoch = 1:epochs
  if end_training
    break;
  end
  fprintf(1, 'Epoch %d\n', epoch);
  this_chunk_CE = 0;
  count = 0;

  % Shuffle the training data.
  rnd_indices = randperm(train_set_size);
  train_input = train_input(:, rnd_indices);
  train_target = train_target(:, rnd_indices);

  % LOOP OVER MINI-BATCHES.
  for m = 1:numbatches
    input_batch = train_input(:, 1+(m-1)*batchsize : m*batchsize);
    target_batch = train_target(:, 1+(m-1)*batchsize : m*batchsize);

    % FORWARD PROPAGATE.
    % Compute the state of each layer in the network given the input batch
    % and all weights and biases.
    [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
      fprop(input_batch, model);

    % COMPUTE LOSS DERIVATIVE.
    %% Expand the target to a sparse 1-of-K vector.
    expanded_target_batch = expansion_matrix(:, target_batch);
    %% Compute derivative of cross-entropy loss function.
    error_deriv = output_layer_state - expanded_target_batch;

    % MEASURE LOSS FUNCTION.
    CE = -sum(sum(...
      expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
    count =  count + 1;
    this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
    if mod(m, show_training_CE_after) == 0
      fprintf(1, 'Batch %d Train CE %.3f\n', m, this_chunk_CE);
      count = 0;
      this_chunk_CE = 0;
    end

    % BACK PROPAGATE.
    %% OUTPUT LAYER.
    hid_to_output_weights_grad =  hidden_layer_state * error_deriv';
    output_bias_grad = sum(error_deriv, 2);
    back_prop_deriv_1 = (model.hid_to_output_weights * error_deriv) ...
      .* hidden_layer_state .* (1 - hidden_layer_state);

    %% HIDDEN LAYER.
    embed_to_hid_weights_grad = embedding_layer_state * back_prop_deriv_1';
    hid_bias_grad = sum(back_prop_deriv_1, 2);
    back_prop_deriv_2 = model.embed_to_hid_weights * back_prop_deriv_1;

    word_embedding_weights_grad(:) = 0;
    %% EMBEDDING LAYER.
    for w = 1:D
       word_embedding_weights_grad = word_embedding_weights_grad + ...
         expansion_matrix(:, input_batch(w, :)) * ...
         (back_prop_deriv_2(1 + (w - 1) * numhid1 : w * numhid1, :)');
    end
    
    % UPDATE WEIGHTS AND BIASES.
    word_embedding_weights_delta = ...
      momentum .* word_embedding_weights_delta + ...
      word_embedding_weights_grad ./ batchsize + ...
      weight_cost * model.word_embedding_weights;
    model.word_embedding_weights = model.word_embedding_weights...
      - learning_rate * word_embedding_weights_delta;

    embed_to_hid_weights_delta = ...
      momentum .* embed_to_hid_weights_delta + ...
      embed_to_hid_weights_grad ./ batchsize + ...
      weight_cost * model.embed_to_hid_weights;
    model.embed_to_hid_weights = model.embed_to_hid_weights...
      - learning_rate * embed_to_hid_weights_delta;

    hid_to_output_weights_delta = ...
      momentum .* hid_to_output_weights_delta + ...
      hid_to_output_weights_grad ./ batchsize + ...
      weight_cost * model.hid_to_output_weights;
    model.hid_to_output_weights = model.hid_to_output_weights...
      - learning_rate * hid_to_output_weights_delta;

    hid_bias_delta = momentum .* hid_bias_delta + ...
      hid_bias_grad ./ batchsize;
    model.hid_bias = model.hid_bias - learning_rate * hid_bias_delta;

    output_bias_delta = momentum .* output_bias_delta + ...
      output_bias_grad ./ batchsize;
    model.output_bias = model.output_bias - learning_rate * output_bias_delta;

    % VALIDATE.
    if mod(m, show_validation_CE_after) == 0
      fprintf(1, 'Running validation ...');
      CE = evaluate(valid_input, valid_target, model);
      fprintf(1, ' Validation CE %.3f\n', CE);

      if CE > best_valid_CE
        fprintf(1, 'Validation error increasing! Training stopped.\n');
        end_training = true;
        model.epochs = epoch;
        break;
      end
      best_valid_CE = CE;
    end
  end
end
end_time = clock;
diff = etime(end_time, start_time);
fprintf(1, 'Finished training in %.2f seconds\n', diff);
fprintf(1, 'Computing final performance stats ..\n');

CE = evaluate(train_input, train_target, model);
fprintf(1, 'Final Training CE %.3f\n', CE);
model.trainCE = CE;
CE = evaluate(valid_input, valid_target, model);
fprintf(1, 'Final Validation CE %.3f\n', CE);
model.validCE = CE;
CE = evaluate(test_input, test_target, model);
fprintf(1, 'Final Test CE %.3f\n', CE);
model.testCE = CE;
end
