%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The demo for paper "Recommended Keypoint-Aware Tracker: Adaptive
% Real-time Visual Tracking Using Consensus Feature Prior Ranking".
% Ran Duan, Changhong Fu, Erdal Kayacan, Danda Pani Paudel
% Implemented by Ran DUAN (without optimize).
%
% Dear User:
%
% 3 datasets (2 of them are from benchmark v1.0 and 1 youtube video) have
% been put into benchmark folder (you can download more). The bounding box
% will be initialized automatically.
%
% Regarding to the benchmark test, we do not recommended you use this code
% since it is a simplified version. Please contact me (duanran@ntu.edu.sg)
% if you need the complete code for algorithm comparison.
%
% Ran DUAN
% 29-12-2015
% At Nanyang Technological University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initialization
clc;
clear all;
close all;
warning('off','all');

addpath('./Recommendation_System');
addpath('./Functions');

%% read frame data

dataset_table = {'FaceOcc2', 'Coupon', 'highway'};
database_folder = 'benchmark';

for dataset_id = 1:length(dataset_table) % read sequences
    close all;
    dataset_folder = dataset_table{dataset_id};
    fulldir = fullfile(cd,database_folder,dataset_folder,'img');
    addpath(fulldir);
    img_dir = dir([fulldir,'/*.jpg']); % load frame names
    
    %% initial target appearance model
    
    figure(1);
    img = imread(img_dir(1).name);
    imshow(img);
    
    try
        rect = importdata(fullfile(cd,database_folder,dataset_folder,'init_rect.txt')); % if the bounding box is spercified
    catch
        rect_groundtruth = importdata(fullfile(cd,database_folder,dataset_folder,'groundtruth_rect.txt')); % otherwise, use ground-truth
        rect = rect_groundtruth(1,:);
    end
    
    imPatch = imcrop(img,rect);
    
    if size(img,3) > 1,
        img_gray = rgb2gray(img);
        im_target = rgb2gray(imPatch);
    else
        img_gray = img;
        im_target = imPatch;
    end
    
    mincontrast_para = 0.2;
    target_corners = detectFASTFeatures(im_target, 'MinContrast', mincontrast_para);
    
    n = 50; % maximum number of features of target
    
    while target_corners.Count < n && mincontrast_para > 0.01 % detect features as many as we can
        mincontrast_para = 0.8*mincontrast_para;
        target_corners = detectFASTFeatures(im_target, 'MinContrast', mincontrast_para);
    end
    
    target_para = mincontrast_para;
    
    target_corners = target_corners.selectStrongest(n);
    
    target_corners.Location = target_corners.Location + repmat([rect(1)-1,rect(2)-1],target_corners.Count,1);
    
    [Target_Features, Target_Points] = extractFeatures(img_gray, target_corners);
    
    pos = [rect(1)+rect(3)/2 rect(2)+rect(4)/2 1];
    target_size = [rect(3), rect(4)];
    
    
    %% initial learning system
    
    Target_Dictionary.Features = Target_Features.Features;
    Target_Dictionary.Location = Target_Points.Location;
    dictionary_size = size(Target_Dictionary.Location,1);
    
    StaticFeatures = binaryFeatures(Target_Dictionary.Features);
    StaticPoints = Target_Points.Location;
    StaticSize = target_size;
    StaticPos = pos;
    
    ConfidenceFeatures = StaticFeatures;
    ConfidencePoint = StaticPoints;
    ConfidenceSize = target_size;
    
    ConfidencePos = pos;
    confidence_weight = 1;
    
    dictionary_maximum_size = 400;
    feature_link_table = eye(dictionary_size);
    tform.T = eye(3);
    
    previous_tform.T = tform.T;
    minimum_trust_n_paris = n/2;
    reset_flag = 1;
    ranking = 100*ones(dictionary_size,3);
    search_flag = 0;
    
    confidence_rank = ranking;
    static_rank = ranking;
    
    confidence_update_flag = 1;
    
    hold on;
%     plot(Target_Dictionary.Location(:,1),Target_Dictionary.Location(:,2),'y.', 'MarkerSize', 10);
    rectangle('Position',rect,'LineWidth',4,'EdgeColor','g');
    hold off;
    
    %% tracking
    
    for frame = 2:numel(img_dir)
        
        img = imread(img_dir(frame).name);
        
        tic
        
        imPatch = imcrop(img,rect);
        if size(img,3) > 1
            im_candidate = rgb2gray(imPatch);
            img_gray = rgb2gray(img);
        else
            im_candidate = imPatch;
            img_gray = img;
        end
        
        mincontrast_para = target_para;
        candidate_corners = detectFASTFeatures(im_candidate, 'MinContrast', mincontrast_para);
        mincontrast_para = 0.2;
        while candidate_corners.Count < n && mincontrast_para > 0.01
            mincontrast_para = 0.9*mincontrast_para;
            candidate_corners = detectFASTFeatures(im_candidate, 'MinContrast', mincontrast_para);
        end
        
        candidate_corners.Location = candidate_corners.Location + repmat([rect(1)-1,rect(2)-1],candidate_corners.Count,1);
        
        [Candidate_Features, Candidate_Points] = extractFeatures(img_gray, candidate_corners);
        Candidate_Dictionary.Features = Candidate_Features.Features;
        Candidate_Dictionary.Location = Candidate_Points.Location;
        
        indexPairs = matchFeatures(StaticFeatures, Candidate_Features);
        n_correspoundence = size(indexPairs,1);
        original_matched_number = n_correspoundence;
        
        %% three layer matching 
        if original_matched_number < 6 % dictionary model
            indexPairs = matchFeatures(ConfidenceFeatures, Candidate_Features);
            n_correspoundence = size(indexPairs,1);
            if n_correspoundence < minimum_trust_n_paris
                Target_Features = binaryFeatures(Target_Dictionary.Features);
                indexPairs = matchFeatures(Target_Features, Candidate_Features);
                n_correspoundence = size(indexPairs,1);
                reset_flag = 0;
            else % confidence model
                Target_Dictionary.Features = ConfidenceFeatures.Features;
                Target_Dictionary.Location = ConfidencePoint;
                target_size = ConfidenceSize;
                ranking = confidence_rank;
                feature_link_table = eye(size(Target_Dictionary.Features,1));
                pos = ConfidencePos;
                reset_flag = 1;
                confidence_update_flag = 1;
            end
        else % original model
            Target_Dictionary.Features = StaticFeatures.Features;
            Target_Dictionary.Location = StaticPoints;
            target_size = StaticSize;
            ranking = static_rank;
            feature_link_table = eye(size(Target_Dictionary.Features,1));
            pos = StaticPos;
            reset_flag = 1;
            confidence_update_flag = 1;
        end
        
        %% weighted shift
        Target_Dictionary_size = size(Target_Dictionary.Location,1);
        matched_target = Target_Dictionary.Location(indexPairs(:,1),:);
        matched_candidate = Candidate_Dictionary.Location(indexPairs(:,2),:);
        consensusPairs = [];
        outlierPair = [];
        Tinliers = [];
        Cinliers = [];
        Toutliers = [];
        Coutliers = [];
        
        shift_weight = ranking(indexPairs(:,1),2)/sum(ranking(indexPairs(:,1),2));
        
        shift_vector = matched_candidate - matched_target;
        pos_shift = sum([shift_weight,shift_weight].*shift_vector,1);
        
        shift_error = sum(abs(shift_vector - repmat(pos_shift,size(shift_vector,1),1)),2);
        shift_error_mu = mean(shift_error);
        shift_error_sigma = std(shift_error);
        
        inlier_index = find(shift_error(shift_error < shift_error_mu));
        if ~isempty(inlier_index)
            Tinliers = indexPairs(inlier_index,1);
            Cinliers = indexPairs(inlier_index,2);
            consensusPairs = [Tinliers,Cinliers];
        else
            search_flag = 1;
        end
        
        %% error handle
        
        outlier_index = find(shift_error(shift_error > shift_error_mu + shift_error_sigma));
        if ~isempty(outlier_index)
            Toutliers = indexPairs(outlier_index,1);
            Coutliers = indexPairs(outlier_index,2);
            outlierPair = [Toutliers,Coutliers];
        end
        
        if size(Tinliers,1) > 5
            inlier_target = Target_Dictionary.Location(Tinliers,:);
            inlier_candidate = Candidate_Dictionary.Location(Cinliers,:);
            
            std_inlier_target = std(inlier_target,1);
            std_inlier_candidate = std(inlier_candidate,1);
            
            new_scale = std_inlier_candidate./std_inlier_target;
        else
            new_scale = [1,1];
        end
        
        tform.T = eye(3);
        tform.T(3,1:2) = pos_shift;
        
        if ~reset_flag
            if sum(pos_shift) < 40
                previous_tform.T = tform.T;
            else
                tform.T = previous_tform.T;
                search_flag = 1;
            end
        end
        
        register_point = [Target_Dictionary.Location, ones(Target_Dictionary_size,1)];
        register_point = register_point*tform.T;
        Target_Dictionary.Location = register_point(:,1:2);
        
        center_location = repmat(pos(1:2),Target_Dictionary_size,1);
        reference_location = Target_Dictionary.Location - center_location;
        reference_location = reference_location.*repmat(new_scale,Target_Dictionary_size,1);
        Target_Dictionary.Location = reference_location + center_location;
        
        new_size = target_size + 0.5*(new_scale - 1).*target_size;
        
        target_size = new_size;
        pos = pos*tform.T;
        
        matching_confidence = sum(ranking(indexPairs(:,1),2))/sum(ranking(:,2));
        try
            confidence_weight = sum(ranking(consensusPairs(:,1),2))/sum(ranking(indexPairs(:,1),2));
        catch
            confidence_weight = 0;
        end
        %% recommendation system
        [Target_Dictionary, feature_link_table, ranking] = update_link_dable(feature_link_table, Target_Dictionary, Candidate_Dictionary, indexPairs, consensusPairs, outlierPair, dictionary_maximum_size, confidence_weight);
        
        %% undate confidence model
        if confidence_update_flag || matching_confidence > 0.2 && original_matched_number > 3
            ConfidenceFeatures = binaryFeatures(Target_Dictionary.Features);
            ConfidencePoint = Target_Dictionary.Location;
            ConfidenceSize = target_size;
            ConfidencePos = pos;
            confidence_rank = ranking;
            confidence_update_flag = 0;
        end
        
        rect = [pos(1:2) - target_size/2, target_size];
        rect(1:2) = rect(1:2) - 0.05*search_flag*target_size;
        rect(3:4) = rect(3:4) + 0.1*search_flag*target_size;
        rect = fix_rect(img,rect);
        search_flag = 0;
        
        timecost = toc;
        %% plot
        imshow(img);
        hold on;
        text(5, 18, strcat('#',num2str(frame)), 'Color',[0.75 0.75 0], 'FontWeight','bold', 'FontSize',20);
        text(100, 18, strcat('time cost = ', num2str(timecost)),'Color',[0.75 0.75 0], 'FontWeight','bold', 'FontSize',10);
        rectangle('Position',rect,'LineWidth',4,'EdgeColor','g');
%         plot(Target_Dictionary.Location(:,1),Target_Dictionary.Location(:,2),'y.', 'MarkerSize', 10);
%         plot(pos(1),pos(2), 'ro', 'LineWidth', 4);
        hold off;
        drawnow;
    end
    
end
