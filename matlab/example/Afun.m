function y = Afun(x,tflag,A, U, S, V)
if strcmp(tflag,'notransp')
    y = A*x - U*(S*(V'*x));
else
    y = A'*x - V*(S*(U'*x));
end
end