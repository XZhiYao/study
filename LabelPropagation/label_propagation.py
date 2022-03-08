import math
import time

import numpy as np
import pandas as pd
from XieZhiyao_Assignment1 import data_processing as processing

top_k = 7
rbf_sigma = 1.5
epochs = 150
threshold = 0.0001

'''
def fill_P_martix(trainX, P_matrix):
    weight_k = []
    trainX.reset_index(inplace=True, drop=True)
    column = 0
    for index_i, row_i in trainX.iterrows():  # len:630
        for index_j, row_j in trainX.iterrows():
            # print('index_i: ', row_i[0], row_i[1])
            # print('index_j: ', row_j[0], row_j[1])
            diff = np.power(row_i[0] - row_j[0], 2) + np.power(row_i[1] - row_j[1], 2)
            weight = np.exp(-(diff / np.power(rbf_sigma, 2)))
            weight_k.append(weight)
            # print('weight: ', weight)
        #
        # top_k_idx = np.array(weight_k).argsort()[::-1][0:top_k]
        # # print('top_k_idx: ', top_k_idx)
        # # print('ALL top_k_value:', sorted(weight_k)[-7::])
        # # for k in top_k_idx:
        # #    print('top_k:', weight_k[k])
        # weight_sum = sum([weight_k[k] for k in top_k_idx])
        # # print('weight_sum:', weight_sum)
        #
        # for k in top_k_idx:
        #     P_matrix.iloc[index_i, k] = (weight_k[k] / weight_sum)
        # # possible = [weight_k[k] / weight_sum for k in top_k_idx]
        # # print('possible:', possible, 'sum:', np.array(possible).sum())
        #
        # # print('sum:', weight_sum)


        weight_sum = sum(weight_k)
        for k in range(len(weight_k)):
            P_matrix.iloc[index_i, k] = (weight_k[k] / weight_sum)

        weight_k.clear()

    print('P_matrix:', P_matrix)

    return P_matrix
'''
def fill_P_martix(trainX, P_matrix):
    weight_k = []
    trainX.reset_index(inplace=True, drop=True)
    matrix = trainX.to_numpy()
    P_matrix = P_matrix.to_numpy()
    x = rbf_sigma**2
    for index_i, row_i in enumerate(matrix):  # len:630
        for index_j, row_j in enumerate(matrix):
            # print('index_i: ', row_i[0], row_i[1])
            # print('index_j: ', row_j[0], row_j[1])
            diff = (row_i[0] - row_j[0])**2 + (row_i[1] - row_j[1])**2
            weight = math.exp(-(diff /x))
            weight_k.append(weight)
        weight_sum = sum(weight_k)
        print('weight_sum:',weight_sum)
        for k in range(len(weight_k)):
            P_matrix[index_i, k] = weight_k[k]/weight_sum
        weight_k = []

    print('P_matrix:', P_matrix)

    return P_matrix

def train(P_matrix, F_matrix, Label_matrix, label_len):

    P = np.array(P_matrix, dtype=float)
    F = np.array(F_matrix, dtype=float).T
    P[np.isnan(P)] = 0
    '''
    Pul = P[label_len:, :label_len]
    Puu = P[label_len:, label_len:]
    fu = np.array(F[label_len:])
    yl = np.array(F[:label_len])

    propagation = True
    last_fu = fu
    round = 0
    while propagation is True:
        round += 1
        fu = np.dot(Puu, fu) + np.dot(Pul, yl)
        loss = np.mean(last_fu - fu)
        print('loss:', loss)
        if loss == 0:
            propagation = False
            continue
        last_fu = fu

    print("迭代次数:", round, "loss:", loss)
    loss_matrix = np.zeros((fu.shape[0], fu.shape[1]))
    for category, row in enumerate(fu):
        index = np.argmax(row)
        loss_matrix[category, index] = 1
    return loss_matrix
    '''
    Label = np.array(Label_matrix, dtype=float).T
    for epoch in range(epochs):
        loss = 0
        for category, column in enumerate(F):
            for num, row in enumerate(P):
                prediction_F = np.dot(column, row)
                '''
                (7,630)
                (1,630)
                (1,630)
                '''
                if num >= label_len:
                    F[category, num] = prediction_F
                    loss += np.abs(prediction_F - Label[category, num])

        pd.DataFrame(F).to_csv('./generate_data/F_update.csv', index=False)
        print("迭代次数:", (epoch + 1), "loss:", loss)
        loss_matrix = np.zeros((F.shape[1], F.shape[0]))
        for category, row in enumerate(F.T):
            index = np.argmax(row)
            loss_matrix[category, index] = 1
        if loss <= threshold:
            break
    return loss_matrix


if __name__ == '__main__':
    df_Aggregation = processing.read_txt('./dataset/Aggregation.txt')
    # processing.draw_originalPic(df_Aggregation)
    trainX, testX, trainY, testY = processing.shuffle_div(df_Aggregation.iloc[:, 0:2], df_Aggregation.iloc[:, 2], 0.2)

    trainX.to_csv("./generate_data/trainSetX.csv", index=False)
    trainY.to_csv("./generate_data/trainSetY.csv", index=False)
    '''
    testX.to_csv("./generate_data/testSetX.csv", index=False)
    testY.to_csv("./generate_data/testSetY.csv", index=False)
    '''
    label_data_X, unlabel_data_X, label_x, label_for_loss = processing.shuffle_div(trainX, trainY, 0.5)
    label_data_X.to_csv("./generate_data/YL.csv", index=False)
    unlabel_data_X.to_csv("./generate_data/YU.csv", index=False)
    label_x.to_csv("./generate_data/YL_label.csv", index=False)
    label_for_loss.to_csv("./generate_data/YU_label.csv", index=False)

    # processing.draw_originalPic(df_Aggregation)
    P_matrix, YL_matrix, YU_matrix, F_matrix, Label_matrix = processing.create_matrix(df_Aggregation, trainX, label_x, label_for_loss, label_data_X,
                                                                        unlabel_data_X)
    # print('len YL:',len(YL_matrix))
    # F = YL + YU
    # P = (L+U) * (L+U)
    # trainX = L+U
    trainX = pd.concat([label_data_X, unlabel_data_X])
    P2_matrix = fill_P_martix(trainX, P_matrix)
    loss_matrix = train(P2_matrix, F_matrix, Label_matrix, len(YL_matrix))
    print(loss_matrix)
