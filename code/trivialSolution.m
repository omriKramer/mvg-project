function [R12,t12]=trivialSolution(points1,points2,K1,K2)
%8 point:
M=buildM( pflat(K1\points1),pflat(K2\points2) );
[~,~,v]=svd(M);
v=v(:,end);
E=reshape(v,3,3);
[u,~,v]=svd(E);
E=u*diag([1,1,0])*v';

%Verify determinant sign:
[U,~,V]=svd(E);
if det(U*V')<1
    E=-E;
    [U,~,V]=svd(E);
end


%4 decompositions:
W=[0 -1 0;1 0 0; 0 0 1];
u3=U(:,3);
P2s={};
P2s{1}=[U*W*V' u3];
P2s{2}=[U*W*V' -u3];
P2s{3}=[U*W'*V' u3];
P2s{4}=[U*W'*V' -u3];

%Choose the solution with maximal potitive depths:
P1=[eye(3) zeros(3,1)];
numPositive=zeros(4,1);
for i=1:4
    [ Xsa ] = triangulate(P1,P2s{i},pflat(inv(K1)*points1),pflat(inv(K2)*points2));
    Xsa=pflat(Xsa);
    p1a=P1*Xsa;
    p2a=P2s{i}*Xsa;
    numPositive(i)=sum(p1a(3,:)>0)+sum(p2a(3,:)>0) ;
end
[~,ind]=max(numPositive);

R12=P2s{ind}(1:3,1:3)';
t12=R12*P2s{ind}(1:3,4);


end

function [ M ] = buildM( points1,points2 )
n=size(points1,2);
M=zeros(n,9);
for i=1:n
    cur=points2(:,i)*points1(:,i)';
    cur=cur(:);
    M(i,:)=cur';
end

end

