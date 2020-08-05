function [P, X_reduced, K] =  KPCA_LINEAR(x)
% find the size of train data
[N, ~] = size(x);  
% linear kernel method
K = x * x';
% centralize data
l = ones(N);
K_centralized = K - l*K/N - K*l/N + l*K*l/(N*N);
% find the eigenvector and eigenvalue by using built in function
[eigen_vector, eigen_value] = eig(K_centralized);
% sort the eigenvector
[value, rank_idx] = sort(diag(eigen_value), 'descend');
vector = eigen_vector(:, rank_idx);
vector = vector(:, 1 : 256);
% construct the projection matrix
value = value(1 : 256);
P = vector ./ sqrt(value');
%find the reduced train X data
X_reduced = K_centralized * P;
end
