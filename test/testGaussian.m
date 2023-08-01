clc; clearvars;
%% Test gaussiano node

%% Create the features

% load gdf and classifier
disp('Load the gdf and the classifier')
[s,h] = sload('/home/paolo/subjects/s12/20230310/exponential_symmetric/c8.20230310.122436.online.mi_bhbf.exponential_sym.gdf');
load('/home/paolo/subjects/s12/c8_bhbf_20230310.smr.mat');

nchannels = settings.acq.channels_eeg;
mask = settings.modules.smr.laplacian;

bands = settings.bci.smr.bands;
gau_M = settings.bci.smr.gau.M;
gau_C = settings.bci.smr.gau.C;

signals = s(:,1:nchannels);

% apply laplacian
lap_signals = signals * mask;
% apply pwelch
[matlab_psd, f] = (proc_spectrogram(lap_signals, 0.5, 0.0625, 0.25, 512));
matlab_psd = log(matlab_psd);
nfreqs = size(matlab_psd, 2);
nwindow = size(matlab_psd,1);

% extract interested frequencies
allFeatures = [];
for c_window = 1:size(matlab_psd, 1)
    features = [];
    for id_chan = 1:length(bands)
        c_bands = bands{id_chan};
        if ~isempty(c_bands)
            for c_b = c_bands
                idx_band = find(c_b == 0:2:256);
                features = cat(1, features, matlab_psd(c_window,idx_band,id_chan));
            end
        end
    end
    allFeatures = cat(1, allFeatures, features);
end
allFeaturesShaped = reshape(allFeatures, 6, size(allFeatures,1)/6)';

%% Save features in a file
disp('save the features')
writematrix(allFeaturesShaped, '/home/paolo/rosneuro_ws/src/rosneuro_decoder_gaussian/test/features.csv');

%% Do raw probability with matlab
allRawProbs = [];

% iterate over windows
for idx_dfet = 1:size(allFeaturesShaped,1)
    dfet = allFeaturesShaped(idx_dfet,:);
    % calculate the raw probability
    [a, raw_prob] = gauClassifier(gau_M, gau_C, dfet); % we knwo bhbf
    [max_value, idMax] = max(raw_prob);

    allRawProbs = cat(1, allRawProbs, raw_prob);
end

%% load file and check if same raw probabilities
disp('load the computed raw probability of rosneuro gaussian decoder')
load('/home/paolo/rosneuro_ws/src/rosneuro_decoder_gaussian/test/rawprobRosneuro.csv');

diff = max(abs(rawprobRosneuro(159,:) - allRawProbs(159,:)), [],'all');
disp(['Max difference in the raw probabilities: ' num2str(diff)]);