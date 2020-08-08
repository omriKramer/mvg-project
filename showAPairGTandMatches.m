clear
close all
addpath(genpath('code'))
pair_path='TrainingSets/pumpkin/1';
% pair_path='TrainingSets/kings_college_front/2000';

load(sprintf('%s/GT/GT.mat',pair_path))
load(sprintf('%s/inputs/data.mat',pair_path))

im1=imread(sprintf('%s/inputs/im1.jpg',pair_path));
im2=imread(sprintf('%s/inputs/im2.jpg',pair_path));

%Read numpy files:
GT_t12np=readNPY(sprintf('%s/GT/GT_t12.npy',pair_path));GT_R12np=readNPY(sprintf('%s/GT/GT_R12.npy',pair_path));
K1np=readNPY(sprintf('%s/inputs/K1.npy',pair_path));K2np=readNPY(sprintf('%s/inputs/K2.npy',pair_path));
points1np=readNPY(sprintf('%s/inputs/points1.npy',pair_path));points2np=readNPY(sprintf('%s/inputs/points2.npy',pair_path));
%Verify that numpy and matlab contains the same information. 
norm(points1-points1np)+norm(points2-points2np)+norm(GT_t12np-GT_t12)+norm(GT_R12np-GT_R12)+norm(K1-K1np)+norm(K2np-K2)


figure, imshow(im1),title('Image 1')
figure, imshow(im2),title('Image 2')

%Plot matches and epipolar lines:
E12=getCrossM(GT_t12)*GT_R12;
F12=inv(K1)'*E12*inv(K2);
plotLines(points1,points2,im2,F12); title('Epipolar Lines')
showMatches(points1,im1,points2,im2);title('Matches')


%Plot camera and points
P1=K1*[eye(3) zeros(3,1)];
P2=K2*[GT_R12' GT_R12'*GT_t12];
t2=pflat(null(P2));
t1=pflat(null(P1));
[ Xsa ] = triangulate((K1)\P1,(K2)\P2,pflat((K1)\points1),pflat((K2)\points2));
Xsa=pflat(Xsa);
figure;
pcshow(Xsa(1:3,:)','MarkerSize',20)
hold on;
%camera 2 in blue
plotCamera('Location',t2(1:3),'Orientation',GT_R12','Size',0.1,'Color',[0 0 1]);

%camera 1 in red
plotCamera('Location',t1(1:3),'Orientation',eye(3),'Size',0.1,'Color',[1 0 0]);

title('Relative Pose and Triangulation')

