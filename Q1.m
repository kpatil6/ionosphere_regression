

%% Given generated data by sampling the sinusoidal function with x-[0,1]
N = 200;
sigma = 0.1; % given standard deviation
x = (0:(1/199):1); % generating 200 datasets between 0 and 1
y = sin(2*pi*x); % original sinusoidal function 
figure(1)
plot(x,y)
xlabel("X uniformly distributed over (0,1)");
ylabel("Sin(2*pi*x)");
hold on;

outputfile = "tables1.txt";
% Adding noise to the signal 
y_i = sin(2*pi*x) + sigma*randn(size(x)); 
plot(x,y_i,'o')
legend("Real sin curve","randomly generated sin curve");
hold off;
%data = string(zeros(1,125));
data = [];
%Dataset
X = x'; % changing 1*200 to 200*1
Y = y';
%% Polynomial fitting experiment
%k = 0 
MSE = zeros(1,5); % Declare the list for collecting MSEs
training_set_size = [10,50,100,150,200]; % training set sizes
lambda = [0.01,0.1,0,1,5]; % Lambda array
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2) % loop over different lambda and training set sizes
        k=0;
        perm = randperm(N,training_set_size(j)); % declare randomly selected m training set values
        X1 = X(perm,:);
        Y1 = Y(perm,:);
        m = size(X1,1);
        X1 = ones(m,1); % adding the bias term
        lambda_matrix = eye(k+1); 
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1; % normal equation to find theta
        MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2)); % calculate MSE's
        t = sprintf("k=%d , lambda = %f, training set size = %d, MSE = %f ",k,lambda(i),training_set_size(j),MSE(j))
        
        
    end
    figure(i+1) % plotting the MSE in the diagram
    plot(training_set_size,MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    hold on;
end
% for k = 1
MSE = zeros(1,5);

for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=1;
        perm = randperm(N,training_set_size(j));
        X1 = X(perm,:);
        Y1 = Y(perm,:);
        m = size(X1,1);
        X1 = [ones(m,1),X1];
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        sprintf("k=%d , lambda = %f, training set size = %d, MSE = %f \n",k,lambda(i),training_set_size(j),MSE(j))
    end
    figure(i+1)
    plot(training_set_size,MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    hold on;
end
MSE = zeros(1,5);
% for k = 2
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=2;
        perm = randperm(N,training_set_size(j));
        X1 = X(perm,:);
        Y1 = Y(perm,:);
        m = size(X1,1);
        X1 = [ones(m,1),X1,X1.^2];
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        sprintf("k=%d , lambda = %f, training set size = %d, MSE = %f \n",k,lambda(i),training_set_size(j),MSE(j))
    end
    figure(i+1)
    plot(training_set_size,MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    hold on;
end
% for k = 3
MSE = zeros(1,5);
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=3;
        perm = randperm(N,training_set_size(j));
        X1 = X(perm,:);
        Y1 = Y(perm,:);
        m = size(X1,1);
        X1 = [ones(m,1),X1,X1.^2,X1.^3];
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        sprintf("k=%d , lambda = %f, training set size = %d, MSE = %f \n",k,lambda(i),training_set_size(j),MSE(j))
    end
    figure(i+1)
    plot(training_set_size,MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    hold on;
end
% for k=9
MSE = zeros(1,5);
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=9; % model
        perm = randperm(N,training_set_size(j)); % generating random positions
        X1 = X(perm,:); % randomly selecting training set size values from X
        Y1 = Y(perm,:); 
        m = size(X1,1); % training set size
        X1 = [ones(m,1),X1,X1.^2,X1.^3,X1.^4,X1.^5,X1.^6,X1.^7,X1.^8,X1.^9]; % generating your X equation
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        sprintf("k=%d , lambda = %f, training set size = %d, MSE = %f \n",k,lambda(i),training_set_size(j),MSE(j))
    end
    figure(i+1)
    plot(training_set_size,MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    legend("k=0","k=1","k=2","k=3","k=9")
    hold off;
end



% Plotting k=9 and lambda=0 to perfectly fit our training dataset
k=9;
X1 = X(1:200,:);
Y1 = Y(1:200,:);
m = size(X1,1);
X1 = [ones(m,1),X1,X1.^2,X1.^3,X1.^4,X1.^5,X1.^6,X1.^7,X1.^8,X1.^9];
lambda_matrix = eye(k+1);
lambda_matrix(1) = 0;
thetainv = pinv(X1'*X1 + lambda(3)*lambda_matrix);
theta = thetainv * X1'*Y1;

figure(10)
title("For K=9 and lambda=0 the polynomial fit")
plot(x,y_i,'o')
hold on
plot(X,X1*theta)
hold off
legend("Given training dataset","Trained polynomial fitting curve")




%% 1.3 dataset

Ntest = 199;
xtest = x + (1/(2*(N-1)));
xtest = xtest(:,1:Ntest);
ytest = sin(2*pi*xtest);

xtest = xtest';
ytest = ytest';


train_MSE = zeros(1,5);
test_MSE = zeros(1,5);
% for k = 2
xtest1 = [ones(199,1),xtest,xtest.^2];
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=2;
        perm = randperm(N,training_set_size(j));
        X1 = X(perm,:);
        Y1 = Y(perm,:);
        m = size(X1,1);
        X1 = [ones(m,1),X1,X1.^2];
        
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        
        train_MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        test_MSE(j) = ((1/199)*sum((xtest1*theta-ytest).^2));
                
    end
    figure(i+10)
    plot(training_set_size,train_MSE)
    hold on
    plot(training_set_size,test_MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    hold on
end


% for k=9
train_MSE = zeros(1,5);
test_MSE = zeros(1,5);

xtest2 = [ones(199,1),xtest,xtest.^2,xtest.^3,xtest.^4,xtest.^5,xtest.^6,xtest.^7,xtest.^8,xtest.^9];
for i=1:size(lambda,2)
    for j=1:size(training_set_size,2)
        k=9; % model
        perm = randperm(N,training_set_size(j)); % generating random positions
        X1 = X(perm,:); % randomly selecting training set size values from X
        Y1 = Y(perm,:); 
        m = size(X1,1); % training set size
        X1 = [ones(m,1),X1,X1.^2,X1.^3,X1.^4,X1.^5,X1.^6,X1.^7,X1.^8,X1.^9]; % generating your X equation
        lambda_matrix = eye(k+1);
        lambda_matrix(1) = 0;
        thetainv = pinv(X1'*X1 + lambda(i)*lambda_matrix);
        theta = thetainv * X1'*Y1;
        train_MSE(j) = ((1/training_set_size(j))*sum((X1*theta-Y1).^2));
        test_MSE(j) = ((1/199)*sum((xtest2*theta-ytest).^2));
                
    end
    figure(i+10)
    plot(training_set_size,train_MSE)
    hold on
    plot(training_set_size,test_MSE)
    xlabel("Training Set Size");
    titl = sprintf("lambda=%d",lambda(i));
    ylabel("MSE");
    title(titl);
    legend("k=2 with Training set","k=2 with Testing set","k=9 with Training set","k=9 with Testing set")
    hold off;
end



