function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propagation
X = [ones(m, 1) X];   
forward_propagation=sigmoid(X*Theta1');
forward_propagation = [ones(m, 1) forward_propagation];

%prediction of our hypothesis(H_{theta})
prediction = sigmoid(forward_propagation * Theta2');

%create a new matrix of size (m \times output_size )
k = zeros(m,num_labels);

%map R ----> R^(output_size \times 1)   (Real to Column vector)
for i = 1: m
    k(i,y(i))= 1;
end

%y is mapped from a real to column vector
y = k;

%cost function
J_half = ((-y).*log(prediction)) - ((1-y).*log(1-prediction)); 
J_half = sum(sum(J_half));
J = J_half/m;

%regularized
Theta1_temp = Theta1;
Theta2_temp = Theta2;

%need to remove first column from both our thetas
Theta1_temp(:,1) = [];
Theta2_temp(:,1) = [];

%theta square sums
Theta1_square_sum=sum(sum((Theta1_temp).^2));
Theta2_square_sum=sum(sum((Theta2_temp).^2));

%reg
reg = (Theta1_square_sum + Theta2_square_sum)*(lambda/(2*m));
J = J + reg;
    
% grad ------------------------
%backpropagation without for loop

% a_2 layer and it's derivative
a_2 = sigmoid(X*Theta1');
a_2 = [ones(m, 1) a_2];
a_2_derivative = (a_2).*(1-a_2);

% small delta_3 and small delta_2
delta_3 = prediction - y;
delta_2 = (delta_3*Theta2).*a_2_derivative;

% remove first column of the small delta_2  [no need to remove first column from delta_3 as it has no bias]
delta_2(:,1) = [];


Delta_2 = delta_3' * a_2;
Delta_1 = delta_2' * X;


%partial derivatives unregularized
Theta1_grad = (1/m)*(Delta_1);
Theta2_grad = (1/m)*(Delta_2);


%regularized

%zero the first columns of thetas so we do not effect when regularizing
Theta1(:,1)=zeros(hidden_layer_size,1);
Theta2(:,1)=zeros(num_labels,1);


%partial derivatives regularized
Theta1_grad = Theta1_grad + ((lambda)*Theta1)/m;
Theta2_grad = Theta2_grad + ((lambda)*Theta2)/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
