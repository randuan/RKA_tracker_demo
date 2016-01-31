% algorithm 1: AppearanceUpdating
% input: G, D, F, R, C, W, \alpha
% output: D, G, x(top-N ranked) % \alpha will be updated outside %
function [Target_Dictionary, new_link_table, ranking] = update_link_dable(old_link_table, Target_Dictionary, Candidate_Dictionary, indexPairs, consensusPairs, outlierPair, dictionary_maximum_size, confidence_weight)

d_t = size(old_link_table,1);
d_c = size(Candidate_Dictionary.Features,1);
new_link_table = eye(d_t + d_c);
new_link_table(1:d_t,1:d_t) = old_link_table;

new_Target_Dictionary.Features = [Target_Dictionary.Features;Candidate_Dictionary.Features];
new_Target_Dictionary.Location = [Target_Dictionary.Location;Candidate_Dictionary.Location];

% vote for matched feature
new_link_table(indexPairs(:,2)+d_t, indexPairs(:,1)) = 1;

% vote for consensus group
if ~isempty(consensusPairs) && ~isempty(outlierPair)
    for ii = 1:size(consensusPairs,1)
        new_link_table(consensusPairs(ii,2)+d_t, outlierPair(:,2)+d_t) = 1;
    end
end

if ~isempty(outlierPair)
    new_Target_Dictionary.Features(outlierPair(:,1),:) = [];
    new_Target_Dictionary.Location(outlierPair(:,1),:) = [];
    new_link_table(outlierPair(:,1),:) = [];
    new_link_table(:,outlierPair(:,1)) = [];
end

out = my_voting_system(new_link_table, confidence_weight); % algorithm 2: VotingProcess
out = out';

d = size(out,1);

% update dictionary

if d > dictionary_maximum_size
    ranking = out(1:fix(dictionary_maximum_size),:);
    Target_Dictionary.Features = new_Target_Dictionary.Features(ranking(:,3),:);
    Target_Dictionary.Location = new_Target_Dictionary.Location(ranking(:,3),:);
    new_link_table = new_link_table(ranking(:,3),ranking(:,3));
else
    Target_Dictionary = new_Target_Dictionary;
    ranking = out;
end

return
