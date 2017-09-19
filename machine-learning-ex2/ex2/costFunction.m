function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp = zeros(m,2);
for i = 1:m
	temp(i,1) = log(sigmoid((theta')*X(i,:)'));
	temp(i,2) = log(1-sigmoid((theta')*X(i,:)'));
endfor
J = (1/m)*(ones(1,m)*(-y.*temp(:,1)-(1-y).*temp(:,2)));

sum = 0;

for j = 1:length(theta)
	for i = 1:m
		grad(j,1) = (sigmoid(theta'*X(i,:)')-y(i))*X(i,j) + grad(j,1);
	endfor
	grad(j,1) = grad(j,1)*(1/m);
endfor
% =============================================================

end
