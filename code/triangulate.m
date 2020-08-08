
function [ Xs ] = triangulate( P1,P2,points1,points2 )
n=size(points1,2);
P1=P1/norm(P1,'fro');
P2=P2/norm(P2,'fro');

Xs=zeros(4,n);
for i=1:n
    Xs(:,i)=triangulateOne(P1,P2,points1(:,i),points2(:,i));
end

end

function X=triangulateOne(P1,P2,x1,x2)
A=[P1 -x1 zeros(3,1);P2 zeros(3,1) -x2];
[~,~,v]=svd(A);
X=v(1:4,end);
X=pflat(X);
end