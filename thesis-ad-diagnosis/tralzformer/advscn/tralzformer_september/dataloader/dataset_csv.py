import pandas as pd
import numpy as np
import toml

from sklearn.preprocessing import LabelEncoder

class CSVDataset:

    def __init__(self, data_file, cnf_file, mode=0):

        # load data csv
        if isinstance(data_file, str):
            print(data_file)
            df = pd.read_csv(data_file, sep=';')
        else:
            df = data_file

        # load configuration file
        self.cnf = toml.load(cnf_file)

        if 'PTID' in df.columns:
            self.ids = list(df['PTID'])

        df.reset_index(drop=True, inplace=True)

        print('{} are selected.'.format(len(df)))

        # check feature availability in data file
        print('Out of {} features in the configuration file, '.format(len(self.cnf['feature'])), end='')
        tmp = [feat for feat in self.cnf['feature'] if feat not in df.columns]
        print('{} are unavailable in data file.'.format(tmp))

        # check label availability in data file
        print('Out of {} labels in the configuration file, '.format(len(self.cnf['label'])), end='')
        tmp = [lbl for lbl in self.cnf['label'] if lbl not in df.columns]
        print('{} are unavailable in data file.'.format(len(tmp)))

        self.cnf['feature'] = {k:v for k, v in self.cnf['feature'].items() if k in df.columns}
        self.cnf['label'] = {k:v for k, v in self.cnf['label'].items() if k in df.columns}

        # get feature and label names
        features = list(self.cnf['feature'].keys())
        labels = list(self.cnf['label'].keys())

        # omit features that are not present in data_file
        features = [feat for feat in features if feat in df.columns]
        labels = [lbl for lbl in labels if lbl in df.columns]

        # drop columns that are not present in the configuration
        df = df[features + labels]

        # drop rows where ALL features are missing
        df_feat = df[features]
        df_feat = df_feat.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete feature missing.'.format(len(df), len(df) - len(df_feat)))
        df = df[df.index.isin(df_feat.index)]
        df.reset_index(drop=True, inplace=True)

        # drop rows where ALL labels are missing
        df_lbl = df[labels]
        df_lbl = df_lbl.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete label missing.'.format(len(df), len(df) - len(df_lbl)))
        df = df[df.index.isin(df_lbl.index)]
        df.reset_index(drop=True, inplace=True)

        # change np.nan to None
        df.replace({np.nan: None}, inplace=True)

        self.df = df

        # construct dictionaries for features and labels
        self.features, self.labels = [], []
        keys = df.columns.values.tolist()
        for i in range(len(df)):
            vals = df.iloc[i].to_list()
            self.features.append(dict(zip(keys[:len(features)], vals[:len(features)])))
            self.labels.append(dict(zip(keys[len(features):], vals[len(features):])))

        # remove if None
        for i in range(len(self.features)):
            for k, v in list(self.features[i].items()):
                if v is None:
                    self.features[i].pop(k)

        # getting label_fractions
        self.label_fractions = {}
        for label in labels:
            try:
                self.label_fractions[label] = self.df[label].value_counts()[1] / len(self.df)
            except:
                self.label_fractions[label] = 0.3

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    @property
    def feature_modalities(self):
        '''...'''
        return self.cnf['feature']
    
    @property 
    def label_modalities(self):
        '''...'''
        return self.cnf['label']