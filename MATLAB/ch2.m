% Chapter 2

%% Create Table
clear; clc
playerinfo = readtable("bball_players.xlsx");
stats = readtable("bball_stats.xlsx");

%% Indexing
clear; clc
A = magic(3);
% 1. Row-Column
% row and cloumn index number
a0 = A(1, 2); % choose 1
a1 = A(1, [2, 3]); % choose [1, 6]
a2 = A([2, 3], end);
a3 = A([1, 3], [2, 3]);
a4 = A(2, :);

% 2. Linear
% row or column index number
b0 = A(2);  % 1 4 7
            % 2 5 8
            % 3 6 9
b1 = A(end);
b2 = A([2, 4, 4, 7]);   % 3 1 1 6

% 3. Logical: true / false
idx = A >= 5;
c0 = A(idx);

%% Indexing with table
clear; clc
playerinfo = readtable("bball_players.xlsx");
ht = playerinfo(1, 13);
wt = playerinfo(2, 'weight');
ht1 = playerinfo(1:3, [13, 14]);
wt1 = playerinfo(1:3, {'height', 'weight'});

hd = playerinfo{1, 13};
wd = palyerinfo{2, 'weight'};
wd1 = playerinfo.weight(2);

height = playerinfo.height;
idx1 = height >= 80;
tmp = height(idx1);
histogram(tmp, 'BinMethod', 'integers')

%% Categorical Data Type
clear; clc
a = [1, 2, 2, 1, 1, 2, 1];
b = categorical(a, [1, 2], {'M', 'F'});

x = [4, 3, 2, 4, 1, 3, 4];
y = categorical(x, 1:4, {'tiny', 'small', 'big', 'huge'}, 'Ordinal', true);
z = y >= 'big';
level = categories(y);

stats = readtable("bball_stats.xlsx");
stats.lgID = categorical(stats.lgID);
disp(categories(stats.lgID));
tmp = stats.lgID == 'NBL';
num_NBL = nnz(tmp); % number of non zero

%% Grouped Operations
clear; clc
playerinfo = readtable("bball_players.xlsx");
playerinfo.pos = categorical(playerinfo.pos);
idx = playerinfo.pos == 'C';
height = playerinfo.height(idx);
height_mean = mean(height);

height_mean1 = grpstats(playerinfo.height, playerinfo.pos); % default: mean
tmp = categories(playerinfo.pos);
height_std = grpstats(playerinfo.height, playerinfo.pos, @std); % @: function handle

height_mean2 = grpstats(playerinfo, 'pos', @mean, 'DataVars', 'height');

%% Table Properties
clear; clc
playerinfo = readtable("bball_players.xlsx");
tmp = playerinfo.Properties.VariableNames;
playerinfo.Properties.VariableNames(10) = {'Position'};
playerinfo.Properties.Descriptions = 'Basketball player info';
summary(playerinfo);

%% Merging Data
clear; clc
X = array2table([1, 5; 3, 4], 'VariableNames', {'A', 'B'});
Y = array2table([10, 50; 30, 40], 'VariableNames', {'A', 'B'});
tmp1 = [X; Y];

Y1 = array2table([3, 40], 'VariableNames', {'A', 'C'});
Z1 = join(Y1, X);
Z2 = innerjoin(X, Y1);
Z3 = outerjoin(X, Y1, 'MergeKeys', true);

%% Missing Data
clear; clc
% 1. Missing data: NaN, undefined, <empty>
% 2. Error data: N/A, none, 9999
X = {'A', 'B', 'C', 'D'; 'N/A', 2, 4, 9999; 1, 3, 5, 7; 4, 5, '-', NaN };
xlswrite('C:\class\work\aa.xlsx', X)

Y = readtable('aa.xlsx', 'TreatAsEmpty', {'N/A', '-'});
Y = standardizeMissing(Y, 9999);
tmp = Y.A;
%mean_A = nanmean(tmp);
mean_A = mean(tmp, 'omitnan');

idx = ismissing(Y); % isnan(Y)
c = any(idx);
r = any(idx, 2);
Y1 = Y;
Y2 = Y;
Y1(:, c) = [];
Y2(r, :) = [];

Y3 = rmmissing(Y);  % rmmissing(Y, 2);  => R2016b

%% Normalizing Data
clear; clc
A = [60, 170; 72, 186; 46, 167];
% 0 ~ 1 normalization
% step 1
tmp = A - min(A);   % => R2016b
tmp1 = bsxfun(@minus, A, min(A));
% step 2
rg = range(A);  % max(A) - min(A)
% step 3
A_new = bsxfun(@rdivide, tmp, rg);

A_new1 = zscore(A);