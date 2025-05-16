function [X,Xiter] = fista_lasso2(Y, D, Xinit, opts)

    if ~isfield(opts, 'backtracking')
        opts.backtracking = false;
    end 

    opts = initOpts(opts);
    lambda = opts.lambda;
    beta = opts.beta;

    if numel(lambda) > 1 && size(lambda, 2)  == 1
        lambda = repmat(opts.lambda, 1, size(Y, 2));
    end
    if numel(Xinit) == 0
        Xinit = zeros(size(D,2), size(Y,2));
    end
    %% cost f
    function cost = calc_f(X)
     %   cost = 1/2 *normF2(Y - D*X);
        temp =ift2(ft2(D).*ft2(X));
        cost = 1/2 *normF2(Y - temp)+1/2 *beta*normF2(X);
    end 
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = calc_f(X) + lambda*norm1(X);
        elseif numel(lambda) == numel(X)
            cost = calc_f(X) + norm1(lambda.*X);
        end
    end 
    %% gradient
%     DtD = D'*D;
%     DtY = D'*Y;
    DtD = ift2(ft2(D').*ft2(D));
    DtY = ift2(ft2(D').*ft2(Y));

    function res = grad(X) 
     %   res = DtD*X - DtY;  ||DX-Y||？时域的卷积等于频域的相乘
       res = ift2(ft2(DtD).*ft2(X)) - DtY+beta*X;
    end 
    %% Checking gradient 
    if opts.check_grad
        check_grad(@calc_f, @grad, Xinit);
    end 

    opts.max_iter = 20;%初始设置的500
    %% Lipschitz constant 
%     L = max(eig(DtD));
 L=1;
    %% Use fista 
    [X, Xiter, ~] = fista_general(@grad, @proj_l1, Xinit, L, opts, @calc_F);

end 