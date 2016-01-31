function rect = fix_rect(img, rect)

rect(:,1) = min(max(1,rect(:,1)),size(img,2));
rect(:,2) = min(max(1,rect(:,2)),size(img,1));
rect(:,3) = max(1, min(size(img,2),rect(:,1)+rect(:,3))-rect(:,1));
rect(:,4) = max(1, min(size(img,1),rect(:,2)+rect(:,4))-rect(:,2));

return