function [P, X_reduced, K] =  KPCA_RBF(X_train, para)
% find the size of train data
[N, ~] = size(X_train);
x_tmp = sum(X.^2, 2);
% RBF kernel method, function reference shows below
K = exp((bsxfun(@minus,bsxfun(@minus,2*X_train*X_train',x_tmp),x_tmp'))/para^2);
% centralize data
l = ones(N);
K_train = K - l*K/N - K*l/N + l*K*l/(N*N);
% find the eigenvector and eigenvalue by using built in function
[eigen_vector, eigen_value] = eig(K_train);
% sort the eigenvector
[value, rank_idx] = sort(diag(eigen_value), 'descend');
vector = eigen_vector(:, rank_idx);
vector = vector(:, 1 : 256);
value = value(1 : 256);
% construct the projection matrix
P = vector ./ sqrt(value');
%find the reduced train X data
X_reduced = K_train * P;
end
% --- The RBF and centralize methods I used are followed the following technique ---
% --- reference: https://zhuanlan.zhihu.com/p/59775730
% --- reference: https://ww2.mathworks.cn/matlabcentral/fileexchange/69378-kernel-principal-component-analysis-kpca
