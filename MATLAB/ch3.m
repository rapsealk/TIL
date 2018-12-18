% Chapter 3

clear; clc
load bbstats.mat

%% 1. Dimensionality Reduction
% 1-1. Multidimensional Scalling
d = pdist(statsnorm);   % nCk: nchoosek(1112, 2)
% cmd: classical multidimensional scalling
% - numeric data
[x, e] = cmdscale(d);   % e: eigen value
figure(1)
pareto(e)
tmp = 100 * e / sum(e);
tmp1 = cumsum(tmp);
tmp1(1:5)

figure(2)
scatter(x(:, 1), x(:, 2))

% md: multidimensional scalling
% - non-numeric data
[x1, e1] = mdscale(d, 2);   % dimension: 2
% 2nd output: stress ~= 0 better

%% 1-2. PCA: Principal Component Analysis
clear; clc
load bbstats.mat

[pc, scrs, vexp, ~, explained] = pca(statsnorm);    % scrs: scores
PCs = array2table(pc, 'RowNames', labels);
disp(cumsum(explained))

figure(1)
scatter3(scrs(:, 1), scrs(:, 2), scrs(:, 3))

%% 1-3: t-Distributed Stochastic Neighbor Embedding (TSNE)
clear; clc
load bbstats.mat

x = tsne(statsnorm);    % default: 2
x1 = tsne(statsnorm, 'NumDimensions', 3);

%% 2. Clustering
% 2-1: k-Means Clustering (centeroid) -> distance
clear; clc
load bbstats.mat

g = kmeans(statsnorm, 2);
g1 = kmeans(statsnorm, 2, 'Start', 'sample');
g2 = kmeans(statsnorm, 2, 'Start', 'cluster');
g3 = kmeans(statsnorm, 2, 'Replicates', 5);

[~, scrs] = pca(statsnorm);
figure(1)
scatter3(scrs(:, 1), scrs(:, 2), scrs(:, 3), 10, g3)

%% 2-2: Gaussian Mixture Models (GMMs) -> stochastic
clear; clc
load bbstats.mat

% gm1 = fitgmdist(statsnorm, 2);
gm1_1 = fitgmdist(statsnorm, 2, 'CovarianceType', 'diagonal');
gm1_2 = fitgmdist(statsnorm, 2, 'RegularizationValue', 0.02);
opts = statset('MaxIter', 500);
gm1_3 = fitgmdist(statsnorm, 2, 'RegularizationValue', 0.02, ...
                                'Replicates', 5, 'Options', opts);

g = cluster(gm1_3, statsnorm);

[pc, scrs] = pca(statsnorm);
figure(1)
scatter(scrs(:, 1), scrs(:, 2), 10, g)

%% 2-3: Interpreting the Groups
clear; clc
load bbstats.mat

g = kmeans(statsnorm, 2);

figure(1);
parallelcoords(statsnorm, 'Group', g, 'Quantile', 0.25);
labelXTicks(labels);

positions = categories(data.pos)
tmp = crosstab(g, data.pos)

%% 2-5: Evaluating Clustering Quality -> Silhouette / evalclusters
clear; clc
load bbstats.mat

opts = statset('MaxIter', 500);
figure(1)
for k = 1:4
    subplot(2, 2, k)
    gm = fitgmdist(statsnorm, k+1, 'Replicates', 5, ...
                            'RegularizationValue', 0.02, 'Options', opts);
    g = cluster(gm, statsnorm);
    silhouette(statsnorm, g);
end

%% 2-6: Hierarchical Clustering
clear; clc

X = [0, 1, 2, 3; 1, 0, 4, 5; 2, 4, 0, 6; 3, 5, 6, 0];
Z = linkage(X);
dendrogram(Z);

g_c1 = cluster(Z, 'maxclust', 3);

load bbstats.mat

idx = data.pos == 'C';
centerstats = zscore(stats(idx, :));

Z = linkage(centerstats);
dendrogram(Z);