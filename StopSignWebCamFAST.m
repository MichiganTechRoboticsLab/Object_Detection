clear all
clc


% read from PC webcam
vidobj = imaq.VideoDevice('winvideo', 1, 'YUY2_640x480');
set(vidobj, 'ReturnedColorSpace', 'rgb');


% Use the newly trained classifier to detect a stop sign in an image.
detector = vision.CascadeObjectDetector('stopSignDetector.xml');
% detector = vision.CascadeObjectDetector();


bbox = [];
% Read a video frame and run the detector.
while isempty(bbox) == 1    
    videoFrame      = step(vidobj);    
    bbox            = step(detector, videoFrame);
end

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.
x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];

% Detect feature points in the face region.
points = detectFASTFeatures(rgb2gray(videoFrame),'MinQuality','0.2', 'ROI', bbox);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);


% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;

i = 0;
while i < 300
    
    
    videoFrame      = step(vidobj);
    
     % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points

        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

        % Apply the transformation to the bounding box
        [bboxPolygon(1:2:end), bboxPolygon(2:2:end)] ...
            = transformPointsForward(xform, bboxPolygon(1:2:end), bboxPolygon(2:2:end));

        % Insert a bounding box around the object being tracked
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon);

        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    else
        
        disp('less than 2')
        videoFrame      = step(vidobj);    
        bbox            = step(detector, videoFrame);
        
        if isempty(bbox) == 0
            % Convert the first box to a polygon.
            % This is needed to be able to visualize the rotation of the object.
            x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
            bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];

            % Detect feature points in the face region.
            points = detectFASTFeatures(rgb2gray(videoFrame), 'ROI', bbox(1,:));

            points = points.Location;

            release(pointTracker)
            initialize(pointTracker, points, videoFrame);

            oldPoints = points;
        end
        
        
    end
    
    imshow(videoFrame)
    
    
    i = i +1;
end

release(vidobj);
clear vidobj;
