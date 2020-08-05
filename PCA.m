% --- initialize data ---
load("usps.csv");
load("usps.t.csv");
y_train = usps(:,1);
X_train = usps(:,2:257);
y_test = usps_t(:,1);
X_test = usps_t(:,2:257);
% --- dimension ---
d = [];
% --- accuracy --- 
accu = [];
% centralize data
Mean = mean(X_train);
X = X_train - repmat(Mean, [size(Mean, 1), 1]);
% find the covariance matrix
Cov = (X'*X);
% find the eigenvector and eigenvalue by using built_in function
[eigen_vector, eigen_value] = eig(Cov);
% sort them
[~, rank_idx] = sort(diag(eigen_value), 'descend');
P = eigen_vector(:, rank_idx);
% find the reduced train and test X data
X_reduced = X * P ;
X_reduced_test = ( X_test - repmat(Mean, [size(X_test, 1), 1]) ) * P;
% dimension from 1 - 256
for i = 1 : 256
    d = [d i];
    % train model by KNN
    mdl = fitcknn(X_reduced(:, 1 : i), y_train, 'NumNeighbors', 1);
    predict_result = predict(mdl, X_reduced_test(:, 1 : i));
    temp = sum(predict_result == y_test) / 2007;
    accu = [accu temp];
end
% output the accuracy
plot(d, accu);
