function [errorR,errort]=evaluateSolutionOnePair(R12_GT,t12_GT,R12_sol,t12_sol)
RMatrixError=R12_GT'*R12_sol;
errorR=norm(rotationMatrixToVector(RMatrixError))*180/pi;

t12_GT=t12_GT/norm(t12_GT);
t12_sol=t12_sol/norm(t12_sol);
errort=acosd(dot(t12_GT,t12_sol));

end