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

axial=true;     
NyquistSampling =0;
res = lambda/(2*NA);       
if(NyquistSampling)
    dxn = lambda/(4*NA);          
    Nn = ceil(N*dx/dxn/2)*2;      
    dxn = N*dx/Nn;                
    oversampling = res/dxn;       
    dk=oversampling/(Nn/2);       
    [kx,ky] = meshgrid(-dk*Nn/2:dk:dk*Nn/2-dk,-dk*Nn/2:dk:dk*Nn/2-dk);
else
    oversampling = res/dx;  
    dk=oversampling/(N/2);      
    [kx,ky] = meshgrid(-dk*N/2:dk:dk*N/2-dk,-dk*N/2:dk:dk*N/2-dk);
end

%% load 2D PSF and OTF
load("PSF.mat")
figure;imagesc(psf2D);colorbar;title('PSF 2D');
otf2D = abs(fftn(psf2D,[N N]));
figure;imagesc(otf2D);colorbar;title('OTF 2D');

%% add aberration
th = atan2(ky, kx);
% Define the aberration coefficient
W40 = 0.0 ;    
W31 = 0.0 ;   
W22 = 0.0 ;   

rho = kr / max(kr(kr <= 1)); 


phi_sph = 2*pi * W40 .* (6*rho.^4 - 6*rho.^2 + 1);

phi_coma = 2*pi * W31 .* (3*rho.^3 - 2*rho) .* cos(th);

phi_ast = 2*pi * W22 .* rho.^2 .* cos(2*th);


phi_total = phi_sph + phi_coma + phi_ast;

pupil_aberr = pupil .* exp(1i * phi_total);

%% Calculate new psf
psf2D_aberr = abs(fftshift(ifft2(pupil_aberr))).^2;
psf2D_aberr = psf2D_aberr * N^2 / sum(abs(pupil_aberr(:)).^2);

figure; imagesc(psf2D_aberr);
colorbar; title('distorted PSF 2D');

otf2D_aberr = abs(fftn(psf2D_aberr, [N N]));
figure; imagesc(otf2D_aberr);
colorbar; title('distorted OTF 2D');


%% load original illumination patterns 
%-----------------------------------------------------
load("Original illumination.mat")
P1 = P3;

%% load distorted illumination patterns 
%-----------------------------------------------------
load("illumination with SA.mat") % load illumination with different aberration

figure(); imagesc(P1(:,:,5)); axis square; axis off; colormap gray; title('pattern');
figure(); imagesc(P1_distorted(:,:,5)); axis square; axis off; colormap gray; title('pattern-distorted');

%% Generate sample
load("obj.mat")
Sample = densite;

P3 = P1;
P1 = P1_distorted;

D1 = zeros(N,N,K,'single');

disp('Generating raw data 1...')

for i = 1:K
    D1(:,:,i) = ifft2(fft2(P1(:,:,i).*Sample).*otf2D);
    D1(:,:,i) = abs(D1(:,:,i));
end

SNR_dB = 30;

signal_power = mean(D1(:).^2);
snr_linear = 10^(SNR_dB/10); 
noise_power = signal_power / snr_linear;


for i = 1:K
    noise = sqrt(noise_power) * randn(N, N); 
    D1(:,:,i) = D1(:,:,i) + noise; 
end

D1 = D1./max(D1(:));
wf_down = mean(D1,3);
figure(); imagesc(mean(D1,3)); axis square; axis off; colormap gray; title('Widefield 2 zoom');


%% calculate output

img = D1;
measureWD = wf_down;
figure;imagesc(img(:,:,1));colorbar;colormap gray;
title('wide field measurement')


%% Wiener deconvolution of widefield image
alpha = 1e-3;
wienerWD = real( ifft2(conj(otf2D).*fft2(measureWD)./(otf2D.*conj(otf2D)+alpha )));
figure;imagesc(wienerWD);title('Wiener filter of wd image');axis square;colormap gray;
 
 %% FISTA Filter
 disp('Blind MSIM deconvolution...');
Y = img;
D = psf2D;

opts.pos = true;opts.lambda = 0.01;
opts.beta = 0.005;
opts.backtracking = false;
nums=224;
temp4_1=zeros(N);
temp5 = zeros(N,N,224);
Xiter=0;

tic;
t=0;
for i=1:nums
    [temp4,Xiter] = fista_lasso(Y(:,:,i), D, [], opts);
    temp5(:,:,i)=abs(temp4);
    t=t+1;
end
times=toc;
times
temp4=sum(temp5,3)/nums;
temp6=zeros(N);
recon2=temp6;
for j=1:nums
    temp6 = (temp5(:,:,j) - temp4).^2;
    recon2 = recon2+temp6;
end
recon2=recon2./nums;
recon2=sqrt(recon2);

figure;imagesc(temp4); axis square; colormap gray; title('temp4');


%% plot and save      

temp = lambda/dx;    
R2 = 5/(NA*pi);
r = 60/temp;
Axisx2 = 1/temp:(1/temp):(60/temp); 
Axisy2 = Axisx2;

SNR = SNR_dB;

folderPath = sprintf('Result_simulator/Aberration/SphericalAberration_obj/NA=%.2f', NA);

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

filename = fullfile(folderPath, sprintf('Original illumination.mat'));
save(filename, 'P3');
figure;imagesc(P3(:,:,5));colormap gray; axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('Original_illumination_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('Original_illumination_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);


filename = fullfile(folderPath, sprintf('illumination with SA.mat'));
save(filename, 'P1_distorted');
figure;imagesc(P1_distorted(:,:,5));colormap gray; axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('Distorted_illumination_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('Distorted_illumination_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);


filename = fullfile(folderPath, sprintf('Sample_SNR%d.mat', SNR));
save(filename, 'Sample');
figure;imagesc(Sample);colormap gray; axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('Sample_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('Sample_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);

filename = fullfile(folderPath, sprintf('wf_SNR%d.mat', SNR));
save(filename, 'measureWD');
figure;imagesc(measureWD);colormap gray; axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('wf_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('wf_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);

filename = fullfile(folderPath, sprintf('wiener_SNR%d.mat', SNR));
save(filename, 'wienerWD');
figure;imagesc(wienerWD);colormap gray;axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('wiener_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('wiener_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);


filename = fullfile(folderPath, sprintf('FISTA_SNR%d.mat', SNR));
save(filename, 'recon2');
figure;imagesc(recon2);colormap gray;axis square;
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off';
filename = fullfile(folderPath, sprintf('fista_SNR%d.tif', SNR));
saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('fista.eps'));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);


%% show part of results


folderPath = sprintf('Result_simulator/Aberration/SphericalAberration_obj/NA=%.2f/Northwest', NA);

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

figure;imagesc(Axisx2,Axisy2,Sample(41:100,41:100));colormap gray; axis square;
 h1 = rectangle('Position',[r-R2 r-R2 2*R2 2*R2],'Curvature',[1 1],'EdgeColor','y','LineWidth',2);
 h2 = rectangle('Position', [r-2*R2, r-2*R2, 4*R2, 4*R2], 'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 2);
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off'; 
 filename = fullfile(folderPath, sprintf('raw_SNR%d.tif', SNR));
 saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('raw_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);



figure;imagesc(Axisx2,Axisy2,measureWD(41:100,41:100));colormap gray; axis square;
 h1 = rectangle('Position',[r-R2 r-R2 2*R2 2*R2],'Curvature',[1 1],'EdgeColor','y','LineWidth',2);
 h2 = rectangle('Position', [r-2*R2, r-2*R2, 4*R2, 4*R2], 'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 2);
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off'; 
 filename = fullfile(folderPath, sprintf('wf_SNR%d.tif', SNR));
 saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('wf_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);


figure;imagesc(Axisx2,Axisy2,wienerWD(41:100,41:100));colormap gray;axis square;
 h1 = rectangle('Position',[r-R2 r-R2 2*R2 2*R2],'Curvature',[1 1],'EdgeColor','y','LineWidth',2);
 h2 = rectangle('Position', [r-2*R2, r-2*R2, 4*R2, 4*R2], 'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 2);
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off'; 
 filename = fullfile(folderPath, sprintf('wiener_SNR%d.tif', SNR));
 saveas(gcf,filename);
 fname = fullfile(folderPath, sprintf('wiener_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);



figure;imagesc(Axisx2,Axisy2,recon2(41:100,41:100));colormap gray;axis square;
 h1 = rectangle('Position',[r-R2 r-R2 2*R2 2*R2],'Curvature',[1 1],'EdgeColor','y','LineWidth',2);
 h2 = rectangle('Position', [r-2*R2, r-2*R2, 4*R2, 4*R2], 'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 2);
ax=gca;
ax.XAxis.Visible='off';
ax.YAxis.Visible='off';
ax.ZAxis.Visible='off'; 
 filename = fullfile(folderPath, sprintf('fista_SNR%d.tif', SNR));
 saveas(gcf,filename); 
 fname = fullfile(folderPath, sprintf('fista_SNR%d.eps', SNR));
 print(gcf,'-depsc', fname);
     unix(['epstopdf ', fname]);