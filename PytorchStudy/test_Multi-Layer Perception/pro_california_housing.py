import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    housedata = fetch_california_housing()
    print('housedata:\n', len(housedata.target))
    X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target, test_size=0.3, random_state=42)
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.transform(X_test)

    housedata_df = pd.DataFrame(data=X_train_s, columns=housedata.feature_names)
    housedata_df['target'] = y_train
    print('housedata_df:\n', len(housedata_df.values))
    datacor = np.corrcoef(housedata_df.values, rowvar=0)
    datacor = pd.DataFrame(data=datacor, columns=housedata_df.columns, index=housedata_df.columns)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(datacor, square=True, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu",
                     cbar_kws={"fraction": 0.046, "pad": 0.03})
    plt.show()

