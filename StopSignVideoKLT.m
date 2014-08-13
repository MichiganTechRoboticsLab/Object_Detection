clear all
clc


% Use the newly trained classifier to detect a stop sign in an image.
detector = vision.CascadeObjectDetector('stopSignDetector.xml');


bbox = [];
% Read a video frame and run the detector.
videoFileReader = vision.VideoFileReader('Stop1_1.mp4');
while isempty(bbox) == 1
    videoFrame      = step(videoFileReader);
    bbox            = step(detector, videoFrame);
end
% % Draw the returned bounding box around the detected sign.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Stop Sign');


% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.
x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];

% Detect feature points in the face region.
% points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
points = detectFASTFeatures(rgb2gray(videoFrame), 'ROI', bbox(1,:));


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

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',[0 0 videoInfo.VideoSize]);

% Track the sign over successive video frames until the video is finished.
Time = 0; count = 0;
while count <220 %~isDone(videoFileReader)

    count = count + 1;
    t = cputime;
    
    videoFrame      = step(videoFileReader);
    
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
        
        
        videoFrame      = step(videoFileReader);    
        bbox            = step(detector, videoFrame);
        
        if isempty(bbox) == 0
            % Convert the first box to a polygon.
            % This is needed to be able to visualize the rotation of the object.
            x = bbox(1, 1); y = bbox(1, 2); w = bbox(1, 3); h = bbox(1, 4);
            bboxPolygon = [x, y, x+w, y, x+w, y+h, x, y+h];

            % Detect feature points in the face region.
%             points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(1,:));
            points = detectFASTFeatures(rgb2gray(videoFrame), 'ROI', bbox(1,:));

            points = points.Location;

            release(pointTracker)
            initialize(pointTracker, points, videoFrame);

            oldPoints = points;
        end
        
        
    end
    
    e = cputime - t;
    Time = Time + e;    
    
    % Display the annotated video frame using the video player object
%     step(videoPlayer, videoFrame);
    imshow(videoFrame)
    mov(count) = getframe;
     
end

TimeAvg = Time/count;
movie2avi(mov(2:end), 'StopSign_KLT2.avi', 'compression', 'None');

% Release resources
release(videoFileReader);
release(videoPlayer);
