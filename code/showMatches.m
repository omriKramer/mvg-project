function im=showMatches(xs1o,im1,xs2o,im2)
%SHOWMATCHES Summary of this function goes here
%   Detailed explanation goes here
figure,imshow(cat(2,im1,im2));
hold on;
for i=1:1:size(xs1o,2)
    yiq=[75,75*(xs1o(1,i)/(size(im1,2))-0.5),75*(xs1o(2,i)/(size(im1,1))-0.5)];
    RGB=ntsc2rgb(yiq);
    plot(xs1o(1,i),xs1o(2,i),'*','color',RGB);
    plot(xs2o(1,i)+size(im1,2),xs2o(2,i),'*','color',RGB);
end
f=gcf;
fr=getframe(f);
im=fr.cdata;
end

