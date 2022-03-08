import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

"""
    构造需要的矩阵和洗牌
"""


def read_txt(path):
    df = pd.read_csv(path, header=None, delimiter="\t")
    return df


def shuffle_div(data, labels, scaling):
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=scaling, random_state=None)
    return trainX, testX, trainY, testY


def draw_originalPic(df):
    df_x = df.iloc[:, 0]
    df_y = df.iloc[:, 1]
    df_label = df.iloc[:, 2]
    fig_df = plt.figure()
    plt.title("Oringinal data of Aggregation")
    ax_df = fig_df.add_subplot(1, 1, 1)
    ax_df.scatter(df_x, df_y, c=df_label, marker='.', cmap='coolwarm')
    plt.show()


def create_matrix(df, trainX, label_x, label_for_loss, label_data_X, unlabel_data_X):
    # numpy.array的构造方法
    # affinity_matrix = np.zeros((num_samples, num_samples), np.float32)

    P_matrix = pd.DataFrame(columns=[i for i in range(len(trainX))], index=[i for i in range(len(trainX))])
    num = np.unique(df.iloc[:, 2])

    label_category = np.zeros((len(label_data_X), len(num)))
    label_category_loss = np.zeros((len(unlabel_data_X), len(num)))
    for row_index in range(len(label_category)):
        # print('row_index:', np.array(label_x)[row_index])
        label_category[row_index, np.array(label_x)[row_index] - 1] = 1
    for row_index in range(len(label_category_loss)):
        label_category_loss[row_index, np.array(label_for_loss)[row_index] - 1] = 1

    # unlabel_category = np.zeros((len(unlabel_data_X), len(num)))
    unlabel_category = np.random.uniform(0, 0, (len(unlabel_data_X), len(num)))
    # print('category:', category)
    YL_matrix = pd.DataFrame(data=label_category, columns=[i for i in range(len(num))], index=[i for i in range(len(label_data_X))])
    YU_matrix = pd.DataFrame(data=unlabel_category, columns=[i for i in range(len(num))], index=[i for i in range(len(unlabel_data_X))])
    Loss_Label_matrix = pd.DataFrame(data=label_category_loss, columns=[i for i in range(len(num))], index=[i for i in range(len(unlabel_data_X))])
    F_matrix = pd.concat([YL_matrix, YU_matrix])
    Label_matrix = pd.concat((YL_matrix, Loss_Label_matrix))

    return P_matrix, YL_matrix, YU_matrix, F_matrix, Label_matrix


'''
if __name__ == '__main__':
    df_Aggregation = read_txt('./dataset/Aggregation.txt')
    draw_oringinalPic(df_Aggregation)

    
    df_Compound = read_txt('./dataset/Compound.txt')
    df_D31 = read_txt('./dataset/D31.txt')
    df_flame = read_txt('./dataset/flame.txt')
    df_jain = read_txt('./dataset/jain.txt')
    df_pathbased = read_txt('./dataset/pathbased.txt')
    df_R15 = read_txt('./dataset/R15.txt')
    df_spiral = read_txt('./dataset/spiral.txt')
    
'''
