function [Y, labels] = Y_Initialize(n, c)
labels = 1:c;
seed = 2024;
rng(seed, 'twister')
labels = [labels, randi(c, 1, n - c)];
labels = labels(randperm(n));
Y = ind2vec(labels)';
Y = full(Y);
labels = labels';
end