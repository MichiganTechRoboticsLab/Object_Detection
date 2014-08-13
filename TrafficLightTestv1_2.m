clear all
clc

% Read image
I = imread('TL34.jpg');

% convert from RGB color space to gray scale
Igray = rgb2gray(I);

% threshold the image
% thr = 0.5 * 255;
thr = mean2(Igray);
BW = Igray > thr;

% reduce the number of connected elemnets using a median filter
BW = medfilt2(BW,[3 3]);

imshow(BW)

% Find the connected components in the BW image (8-connected neighborhood)
CC = bwconncomp(BW);
ObjIdxs = CC.PixelIdxList;

% Find objects that have a round(ish) shape.
Mask = false(size(Igray)); % overall mask (init.)
count = 0; % counter of round objects
for i = 1 : length(ObjIdxs)
    
    
    CurrentObjIdxs = cell2mat(ObjIdxs(i));
    PixCount = length(CurrentObjIdxs); % number of pixcles in each object
    
    
    if  PixCount > 3 && PixCount < 100
        
        [R,C] = ind2sub(size(Igray),CurrentObjIdxs);
        width = max(C)-min(C) + 1;
        hight = max(R)-min(R) + 1;
        
        if (max(width,hight) < 2*min(width,hight)) && (PixCount > 0.75*width*hight) % find round objects (**)                        
            
            
            % find the seed point in each blob, for the region growing algorithm            
            [~,Maxind] = max(Igray(CurrentObjIdxs));
            ExtremaIdx = CurrentObjIdxs(Maxind);    % find the extrema in the image of current obj (blob)
            
            % Region growing algorithm
            [R,C] = ind2sub(size(Igray),ExtremaIdx);
            PartialMask = false(size(Igray));
            [~, tempMask,BlobStatus] = regionGrowing(Igray, [R C],0.5*Igray(ExtremaIdx),[],[],[],[],width*hight);
            if BlobStatus == false
                PartialMask = false(size(Igray));
            else
              BlobInd = find(tempMask);
              [R,C] = ind2sub(size(tempMask),BlobInd);
              width = max(C)-min(C) + 1;
              hight = max(R)-min(R) + 1;
              if (max(width,hight) < 2*min(width,hight)) && (length(BlobInd) > 0.6*width*hight) % find round objects (**)
                  PartialMask = tempMask;
                  count = count + 1;
              end
            end

            Mask = Mask | PartialMask;  % overall mask                                
        end        
            
    end
end


imshow(Igray)
figure
imshow(Mask)

 % (**) 0.75 =(appx) 0.95*(pi/4); where pi/4 is the ratio between the area of a circle and the square enclosing it.