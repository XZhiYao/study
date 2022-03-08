from datetime import datetime
import os
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np
import XieZhiyao_Assignment2 as data

threshold = 0.0001  # final loss
Lamtha = 0.02  # Normolization factor
Gamma = 0.01  # learn rate
epochs = 1500  # number of iteration


def matrixFactorization(R, k):
    P = np.random.normal(-.05, .05, (R.shape[0], k))
    Q = np.random.normal(-.05, .05, (R.shape[1], k))

    return P, Q


def unparellel_fastest_sgd(P, Q, R_hat, R_DF, R_testX, R_testY):
    now = datetime.now()
    print('sgd start:', now)

    Mu = np.nanmean(R_hat)
    R_count = R_hat.size - np.count_nonzero(np.isnan(R_hat))
    b1 = []
    b2 = []
    for row in R_hat:
        b1_row = np.nanmean(row) - Mu
        b1.append(b1_row)
    for column in R_hat.T:
        b2_column = np.nanmean(column) - Mu
        if np.isnan(b2_column):
            b2_column = 0
        b2.append(b2_column)

    for epoch in range(epochs):
        start = datetime.now()

        loss = 0
        pool = Pool(4)
        user, item = np.where(~np.isnan(R_hat))

        for i, j in zip(user, item):

            prediction = np.dot(P[i], Q[j]) + b1[i] + b2[j] + Mu
            error = R_hat[i, j] - prediction
            # print('updata para:')
            b1[i] = b1[i] + Gamma * (error - Lamtha * b1[i])
            b2[j] = b2[j] + Gamma * (error - Lamtha * b2[j])
            P[i] = P[i] + Gamma * (error * Q[j] - Lamtha * P[i])
            Q[j] = Q[j] + Gamma * (error * P[i] - Lamtha * Q[j])

            # print('updata reg:')
            P_Norm = np.power(np.linalg.norm(P[i]), 2)
            Q_Norm = np.power(np.linalg.norm(Q[j]), 2)
            regularization = Lamtha * (np.power(b1[i], 2) + np.power(b2[j], 2) + P_Norm + Q_Norm)

            loss += (np.power((R_hat[i, j] - prediction), 2) + regularization)

        loss = loss / R_count

        if loss <= threshold:
            break
        # w_gradient = (loss + loss * Gamma)

        # evaluate
        RMSE = 0
        # print('R_testX.shape:', R_testX.shape)
        # print('R_textY.shape_max:', max(R_testY['Rating']))
        for i in range(R_testX.shape[0]):
            userID_Test = R_testX.iloc[i, 0]
            # print('1userid:', userID_Test)
            # userID_Test = R_DF.index.get_loc("User" + str(userID_Test))
            userID_Test_index = R_DF.index.get_loc("User" + str(userID_Test))
            # print('row_DE1:', userID_Test)
            # print('row_DF:', userID_Test_index)

            movieID_Test = R_testX.iloc[i, 1]
            # print('1movieid:', movieID_Test)
            movieID_Test_index = R_DF.columns.get_loc("Movie" + str(movieID_Test))
            # print('column_DF:', movieID_Test_index)

            rating_Test = R_testY.iloc[i, 0]
            pre = np.dot(P[userID_Test_index], Q[movieID_Test_index]) + b1[userID_Test_index] + b2[
                movieID_Test_index] + Mu
            RMSE += np.power(rating_Test - pre, 2)
            # print(f'rating:{rating_Test},predict_ratings:{pre},user_id:{userID_Test},movie_id:{movieID_Test},uidindex:{userID_Test_index},midindex:{movieID_Test_index}')

        RMSE = (RMSE / R_testX.shape[0]) ** 0.5
        end = datetime.now()

        print("迭代次数:", (epoch + 1), "loss:", loss, "RMSE:", RMSE, "Time:", end - start)


if __name__ == '__main__':
    print('read pre data:', datetime.now())
    R_DF, R_ARR, R_testX, R_testY = data.dataSetPre()
    print('read pre data end:', datetime.now())
    P, Q = matrixFactorization(R_ARR, 100)

    unparellel_fastest_sgd(P, Q, R_ARR, R_DF, R_testX, R_testY)
