import os.path
from glob import glob
from os import makedirs

import pandas as pd

print('DATA PREPARATION')
files = glob('processing/*.csv') + glob('cons/*.csv') + glob('gc_content/*.hist')

for file in files:
    print(file)
    feature_df = pd.read_table(file, header=None, sep=',')
    print('dim', len(feature_df), 'x', len(feature_df.columns))
    if len(feature_df.columns) != 100:
        print('WE are DOOMED')
        exit(1)

path = 'result'
try:
    makedirs(path)
except:
    pass

features = ['SMC1A', 'PARP1', 'Spi1', 'PAF1', 'SMC3',
            'H4K20me1', 'H3K4me3', 'H3K36me3', 'Rad21', 'H3K9Me3', 'H2AFZ',
            'H3K27Me3', 'STAG1', 'PRKCQ', 'CTCF', 'ERCC6',
            # 'gc', 'cons'
            ]
stains = {
    'gpos25': 0,
    'gpos75': 0,
    'gpos100': 1,
    'gpos50': 0,
    'acen': 0,
    'gvar': 0,
    'stalk': 0,
    'gneg': 0,
}

train_dfs = []
test_dfs = []

for file in files:
    name = os.path.basename(file).split('/')[-1].split('_')
    feature_name, stain_name = name[0], name[-1]
    stain_name = stain_name.split('.')[0]
    if feature_name not in features or stain_name not in stains:
        print('UNKNOWN file', feature_name, stain_name, file)
        continue
    feature_id = features.index(feature_name)
    stain_id = stains[stain_name]
    # print('LOADING feature file', feature_name, stain_name, file)
    feature_df = pd.read_table(file, sep=',',
                               names=['A' + str(k) for k in range(100)])
    feature_df['stain'] = stain_id
    feature_df['feature'] = feature_id
    # 80% to train
    train_number = int(len(feature_df) * .8)
    train_dfs.append(feature_df.loc[range(train_number)])
    # 20% to test
    test_dfs.append(feature_df.loc[range(train_number, len(feature_df))])

train_df = pd.concat(train_dfs)
print('TRAIN dim', len(train_df), 'x', len(train_df.columns))
print(train_df.head(1))
test_df = pd.concat(test_dfs)
print('TEST dim', len(test_df), 'x', len(test_df.columns))
print(test_df.head(1))

test_df.to_csv(path + '/test.csv', sep=',', index=None, header=True)
train_df.to_csv(path + '/train.csv', sep=',', index=None, header=True)
print("TEST/TRAIN preprocessing done")
