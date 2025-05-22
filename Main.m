addpath('func')
addpath('Illumination')

%% Input parameters
%-----------------------------------------------------
Read_out_noise_on = 1;
S = 15; 
K = S^2;
N=200;          
pixelsize = 4;    
magnification = 100; 
dx=pixelsize/magnification;     % Sampling in lateral plane at the sample in um
NA=1.45;         
lambda=0.488;   % Wavelength in um

% load 2D PSF and OTF
%-----------------------------------------------------
load("PSF.mat")
figure;imagesc(psf2D);colorbar;title('PSF 2D');
otf2D = abs(fftn(psf2D,[N N]));
% figure;imagesc(otf2D);colorbar;title('OTF 2D');


% Load illumination patterns
load("Original illumination.mat"); 
Illumination{1} = P3; illum_names{1} = 'Original illumination';
load("illumination with SA.mat");  
Illumination{2} = P1_distorted; illum_names{2} = 'illumination with SA';
load("illumination with Astig.mat"); 
Illumination{3} = P1_distorted; illum_names{3} = 'illumination with Astig';

% Generate sample
disp("Loading sample...")
load("obj.mat")
Sample = densite;

% Reconstruction parameters
SNR_dB = 30;% noise parameters
alpha = 1e-3;% Wiener parameters

% fista parameters
opts.pos = true;
opts.lambda = 0.01;
opts.beta = 0.005;
opts.backtracking = false;
nums = K;

for idx = 1:length(Illumination)
    disp(['Processing: ' illum_names{idx}]);

    P = Illumination{idx};  % use different Illumination
    
    D1 = zeros(N,N,K);
    for i = 1:K
        D1(:,:,i) = ifft2(fft2(P(:,:,i).*Sample).*otf2D);
        D1(:,:,i) = abs(D1(:,:,i));
    end
    signal_power = mean(D1(:).^2);
    snr_linear = 10^(SNR_dB/10); 
    noise_power = signal_power / snr_linear;
    for i = 1:K
        noise = sqrt(noise_power) * randn(N, N); 
        D1(:,:,i) = D1(:,:,i) + noise; 
    end
    D1 = D1 ./ max(D1(:));
    measureWD = mean(D1,3);

    % Wiener
    wienerWD = real(ifft2(conj(otf2D) .* fft2(measureWD) ./ (abs(otf2D).^2 + alpha)));

    % blind
    disp('Blind MSIM deconvolution...')
    Y = D1;
    D = psf2D;
    temp5 = zeros(N,N,nums);
    for i = 1:nums
        [temp4, ~] = fista_lasso2(Y(:,:,i), D, [], opts);
        temp5(:,:,i) = abs(temp4);
    end
    temp4_mean = sum(temp5,3) / nums;
    recon2 = zeros(N);
    for j = 1:nums
        recon2 = recon2 + (temp5(:,:,j) - temp4_mean).^2;
    end
    recon2 = sqrt(recon2 ./ nums);

%     results(idx).name = illum_names{idx};        
%     results(idx).wienerWD = wienerWD;             % Wiener
    results(idx).recon2 = recon2;                 % FISTA
end




%% plot and save      
figure;
pattern1 = subplot(1,3,1);
imagesc(Illumination{1}(:,:,5));colormap gray; axis square;title('Original')
pattern2 = subplot(1,3,2);
imagesc(Illumination{2}(:,:,5));colormap gray; axis square;title('SA')
pattern3 = subplot(1,3,3);
imagesc(Illumination{3}(:,:,5));colormap gray; axis square;title('Astig')

set([pattern1 pattern2 pattern3], 'XColor', 'none', 'YColor', 'none');


figure;
a1 = subplot(2,3,1);
imagesc(Sample);colormap gray; axis square;title("Ground truth")
a2 = subplot(2,3,2);
imagesc(measureWD);colormap gray; axis square;title("Wide-field")
a3 = subplot(2,3,3);
imagesc(wienerWD);colormap gray;axis square;title("wiener")
a4 = subplot(2,3,4);
imagesc(results(1).recon2);colormap gray;axis square;title("proposed-Original")
a5 = subplot(2,3,5);
imagesc(results(2).recon2);colormap gray;axis square;title("proposed-SA")
a6 = subplot(2,3,6);
imagesc(results(3).recon2);colormap gray;axis square;title("proposed-Astig")

set([a1 a2 a3 a4 a5 a6], 'XColor', 'none', 'YColor', 'none');

