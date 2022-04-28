flow_file = './1.mat';
load(flow_file);
img = flowToColor(flow);
figure;
imshow(img)
saveas(gcf,'test','png')