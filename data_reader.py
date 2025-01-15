import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataReader:
    def __init__(self, action, label, splits=[0.6, 0.2, 0.2],
                 path='Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv'):

        self.n_action = 2
        self.name = 'Kevin'

        self.label_col = label

        segment_map = {'No E-Mail': 0, 'Mens E-Mail': 1, 'Womens E-Mail': 2}

        self.actions = [0]
        if 'women' in action:
            self.actions.append(1)
        if 'men' in action:
            self.actions.append(2)

        if len(self.actions) < 2:
            raise ValueError("Action not found. Specify 'women' or 'men' in action.")

        # Load dataset with pandas
        self.data = pd.read_csv(path)

        # Filter rows based on actions & convert actions
        self.data['segment'] = self.data['segment'].map(segment_map)
        self.data = self.data[self.data['segment'].isin(self.actions)]

        numerical_features = ['recency', 'history']
        scaler = StandardScaler()
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        # One-hot encode categorical features
        categorical_features = ['history_segment', 'zip_code', 'channel']
        self.data = pd.get_dummies(self.data, columns=categorical_features, dtype=int)

        # Map segment to actions
        action_map = {'Web': 0, 'Phone': 1, 'Multichannel': 2}
        self.data['action'] = self.data['segment'].map(action_map)

        # Extract features, actions, and labels
        self.features = self.data.drop(columns=['segment', 'spend', 'conversion', 'action', 'visit']).to_numpy()
        self.actions = self.data['segment'].to_numpy()
        self.labels = self.data[self.label_col].to_numpy()

        self.n_feature = self.features.shape[1]

        # Combine data into tuples
        self.datas = list(zip(self.features, self.actions, self.labels))

        # Shuffle and split data
        np.random.shuffle(self.datas)
        n_all = len(self.datas)
        n_train = int(n_all * splits[0])
        n_validate = int(n_all * splits[1])
        n_test = n_all - n_train - n_validate

        self.datas_train = self.datas[:n_train]
        self.datas_validate = self.datas[n_train:n_train + n_validate]
        self.datas_test = self.datas[-n_test:]

    def get_datas(self):
        return self.datas_train, self.datas_validate, self.datas_test

