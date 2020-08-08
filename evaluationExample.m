clear
close all
addpath(genpath('code'))
errorsR=[];
errorst=[];


for i=1:1:500
    pair_path=sprintf('TrainingSets/buddah/%d',i);
    
    load(sprintf('%s/GT/GT.mat',pair_path))
    load(sprintf('%s/inputs/data.mat',pair_path))
    
    [R12_sol,t12_sol]=trivialSolution(points1,points2,K1,K2);
    [errorR,errort]=evaluateSolutionOnePair(GT_R12,GT_t12,R12_sol,t12_sol);
    errorsR=[errorsR;errorR];
    errorst=[errorst;errort];    
end
sprintf('R mean error %f',mean(errorsR))
sprintf('t mean error %f',mean(errorst))
sprintf('R median error %f',median(errorsR))
sprintf('t median error %f',median(errorst))

figure, histogram(errorsR,10)
legend('Trivial')
title('R')

figure, histogram(errorst,10)
legend('Trivial')
title('t')