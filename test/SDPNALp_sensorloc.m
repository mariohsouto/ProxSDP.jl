$ DISCLAIMER
% This test demonstrates SDPNAL failing to solve the
% sensors localization problem (sensorloc), even
% with its own matlab interface. Therefore the errors
% in the benchmark are not due to JuMP

%% problem parameters
n = 100;
m = floor(0.1 * n);

%% Automatically generate problem data

% Sensor true position (2 dimensional)
x_true = rand(2, n);

% Distances from sensors to sensors
d = zeros(n , n);
for i = 1:n
    for j = 1:i
        d(i,j) = norm(x_true(:, i) - x_true(:, j));
    end
end

% Anchor positions
a = rand(2, m);

% Distances from anchor to sensors
d_bar = zeros(m , n);
for i = 1:m
    for j = 1:n
        d_bar(i,j) = norm(x_true(:, j) - a(:, k));
    end
end

%% Initialize SDP problem

mymodel = ccp_model('test_model') ;

X = var_sdp(n+2 ,n+2) ;

mymodel.add_variable(X);

%% Constraint with distances from anchors to sensors
for j = 1:n
    for k = 1:m
        mymodel.add_affine_constraint( (a(1, k)^2) * X(1,1) + (a(2, k)^2)*X(2,2) - 2 * a(1, k)* X(1, j+2)  - 2 * a(2,k) * X(2, j+2)  + X(j+2, j+2) == d_bar(k, j)^2 );
    end
end

%% Constraint with distances from sensors to sensors
count = 0;
count_all = 0;
has_ctr = zeros(n,n);
for i = 1:n
    for j = 1:(i - 1)
        count_all = count_all + 1;
        if rand() > 0.9
            count = count + 1;
            has_ctr(i,j) = 1;
            mymodel.add_affine_constraint(X(i+2,i+2) + X(j+2,j+2) - 2*X(i+2,j+2) == d(i, j)^2);
        end
    end
end

%% Add remaining constraints
mymodel.add_affine_constraint( X(1, 1) == 1.0);
mymodel.add_affine_constraint( X(1, 2) == 0.0);
mymodel.add_affine_constraint( X(2, 1) == 0.0);
mymodel.add_affine_constraint( X(2, 2) == 1.0);

%% set objective
mymodel.minimize( 0 * X(1, 1) ) ;

%% optimize function
mymodel.solve ;
