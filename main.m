%initialize the data set
load("usps.csv");
load("usps.t.csv")
y_train = usps(:,1);
X_train = usps(:,2:257);
y_test = usps_t(:,1);
X_test = usps_t(:,2:257);

% --- To test PCA, you can simply run the PCA file ---
% --- To test different kernel method in KPCA, you can choose to ---
% --- Uncomment the follow code ---

% --- !IMPORTANT! To successfully run the program
% --- you have to change the corresponding Kernel method
% --- in test file, you can simply uncomment the corresponding code

% --- Test Linear Kernel ---
 [P, X_reduced, K] = KPCA_LINEAR(X_train);
 test;

% --- Test Polynomial Kernel ---
% [P, X_reduced, K] =  KPCA_POLY(X_train);
% test;

% --- Test RBF Kernel ---
% [P, X_reduced, K] = KPCA_RBF(X_train, 8);
% test;
