%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for sample weights
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for bias
parameters.alpha_gamma = 1;
parameters.beta_gamma = 1;

%set the hyperparameters of gamma prior used for kernel weights
parameters.alpha_omega = 1;
parameters.beta_omega = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems

%set the number of iterations
parameters.iteration = 200;

%set the margin parameter
parameters.margin = 1;

%determine whether you want to calculate and store the lower bound values
parameters.progress = 0;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the standard deviation of intermediate representations
parameters.sigmag = 0.1;

%initialize the kernels and class labels for training
Ktrain = ??; %should be an Ntra x Ntra x P matrix containing similarity values between training samples
ytrain = ??; %should be an Ntra x 1 matrix containing class labels (contains only -1s and +1s)

%perform training
state = bemkl_supervised_classification_variational_train(Ktrain, ytrain, parameters);

%display the kernel weights
display(state.be.mean(2:end));

%initialize the kernels for testing
Ktest = ??; %should be an Ntra x Ntest x P matrix containing similarity values between training and test samples

%perform prediction
prediction = bemkl_supervised_classification_variational_test(Ktest, state);

%display the predicted probabilities
display(prediction.p);
