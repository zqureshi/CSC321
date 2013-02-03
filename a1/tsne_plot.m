% This function plots the learned embedding in 2-d space using TSNE.
% Usage:
% > tsne_plot(model);
% where model is the output of the training program.
function [mappedX]=tsne_plot(model)
  if size(ver('Octave'),1)
    OctaveMode = 1;
  else
    OctaveMode = 0;
  end

  mappedX = tsne(model.word_embedding_weights);
  if ~OctaveMode
    %scatter(mappedX(:, 1), mappedX(:, 2), 8, 'black', 'filled');
  end
  text(mappedX(:, 1), mappedX(:, 2), model.vocab);
end
