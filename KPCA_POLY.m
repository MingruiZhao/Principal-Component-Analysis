function [P, X_reduced, K] =  KPCA_POLY(X_train)
% find the size of train data
[N, ~] = size(X_train);   
% polynomial kernel method
K = (X_train * X_train' + 1) .^ 2;
% centralize data
l = ones(N);
K_train =K-l*K/N-K*l/N+l*K*l/(N*N);
% find the eigenvector and eigenvalue by using built in function
[eigen_vector, eigen_value] = eig(K_train);
% sort the eigenvector
[value, rank_idx] = sort(diag(eigen_value), 'descend');
vector = eigen_vector(:, rank_idx);
vector = vector(:, 1 : 256);
value = value(1 : 256);
% construct the projection matrix
P = vector ./ (sqrt(value'));
%find the reduced train X data
X_reduced = K_train * P;
end
