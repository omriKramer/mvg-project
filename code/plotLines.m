function  im=plotLines(points1,points2,im2,F12)
%PLOTLINES Summary of this function goes here
%   Detailed explanation goes here
inds=1:round((size(points1,2)/20)):size(points1,2);

ls=F12'*points1(:,inds);

figure, imshow(im2);
hold on;
for i=1:length(inds)
    colorr=rand(3,1);
    rital(ls(:,i),colorr);
    
    plot(points2(1,inds(i)),points2(2,inds(i)),'*y');
    plot(points2(1,inds(i)),points2(2,inds(i)),'o','Color',colorr,'LineWidth',3,'MarkerSize',6);
end
f=gcf;
fr=getframe(f);
im=fr.cdata;
end

