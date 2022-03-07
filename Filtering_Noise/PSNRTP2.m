function err = PSNRTP2( X,Y )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%PSNR
[N, M]=size(X);
X=double(X);
Y=double(Y);
d=0;
for i=1:N
    for j=1:M
       d=d+abs(X(i,j)-Y(i,j)).^2;
    end
end
d=d/(M*N);
err= 10*log10(255^2/d);
end 