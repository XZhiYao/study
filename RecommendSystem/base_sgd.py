from datetime import datetime

import numpy as np
from XieZhiyao_Assignment2 import dataPre as data


def matrixFactorization(R, k):
    # [0] row of matrix; [1] column of matrix

    P = np.random.normal(-.01, .01, (R.shape[0], k))
    Q = np.random.normal(-.01, .01, (R.shape[1], k))

    '''
    P = np.random.rand(R.shape[0], k)
    Q = np.random.rand(R.shape[1], k)
    '''

    '''
    # 直接分解的方法，可以试着用一下
    q, r = np.linalg.qr(R)
    print('p:', q.shape)
    print('q:', r.shape)
    print('q*r:', np.dot(q, r))
    '''

    return P, Q


def SGD(P, Q, R_hat, R_DF, R_testX, R_testY):
    now = datetime.now()
    print('sgd start:', now)
    # Mu = 0
    # calculate the mean value
    Mu = np.nanmean(R_hat)
    # print('Mu:', Mu)
    '''
    for row in R_hat:
        Mu += np.nansum(row)
    Mu = Mu / (R_hat.shape[0] * R_hat.shape[1])

    print('Mu:,hua fei shi jian :', Mu, datetime.now()-now)
    
    b1 = np.zeros()
    for row in R_hat:
        b1 += (Mu - row)  # user bias column
    b1 = b1 / R_hat.shape[1]

    b2 = np.zeros()
    for column in R_hat.T:
        b2 += (Mu - column)
    b2 = b2 / R_hat.shape[0]
    '''

    b1 = []
    b2 = []
    # b = 0

    for row in R_hat:
        b1_row = np.nanmean(row) - Mu
        # (10*Mu - mean*10)/10 == Mu - mean
        # Mu - 1 + Mu -2 + Mu-3 = 3
        # 3.5 - 1
        #  2.5
        '''
        if np.isnan(b1_row):
            print(row)
            import pdb
            pdb.set_trace()
        '''
        b1.append(b1_row)
    # for row in R_hat:
    #     for column in row:
    #         if not np.isnan(column):
    #             b += (Mu - column)
    #         else:
    #             pass
    #     b1.append(b / R_hat.shape[1])
    # print('b1:', b1)
    for column in R_hat.T:
        b2_column = np.nanmean(column) - Mu
        if np.isnan(b2_column):
            b2_column = 0
        b2.append(b2_column)

    # print('b2:', b2)
    # w_gradient = np.zeros(shape=(1, R_hat.shape[1] - 1))
    # print('w_gradient:', w_gradient.shape)

    threshold = 0.0001  # final loss
    Lamtha = 0.02  # Normolization factor
    Gamma = 0.01  # learn rate
    epochs = 1500  # number of iteration
    R_count = R_hat.size - np.count_nonzero(np.isnan(R_hat))
    print('P:', P.shape)
    print('Q:', Q.shape)
    for epoch in range(epochs):

        start = datetime.now()
        loss = 0  # initial loss

        for i, row in enumerate(R_hat):
            for j, R in enumerate(row):

                if not np.isnan(R):
                    startPre = datetime.now()
                    # print('prediction:')
                    prediction = np.dot(P[i], Q[j]) + b1[i] + b2[j] + Mu

                    error = R - prediction
                    # print('updata para:')
                    b1[i] = b1[i] + Gamma * (error - Lamtha * b1[i])
                    b2[j] = b2[j] + Gamma * (error - Lamtha * b2[j])
                    P[i] = P[i] + Gamma * (error * Q[j] - Lamtha * P[i])
                    Q[j] = Q[j] + Gamma * (error * P[i] - Lamtha * Q[j])

                    # print('updata reg:')
                    # P_Norm = np.linalg.norm(P[i])
                    P_Norm = np.power(np.linalg.norm(P[i]), 2)
                    # Q_Norm = np.linalg.norm(Q[j])
                    Q_Norm = np.power(np.linalg.norm(Q[j]), 2)
                    regularization = Lamtha * (np.power(b1[i], 2) + np.power(b2[j], 2) + P_Norm + Q_Norm)
                    endPre = datetime.now()
                    #print('Pre time cost: ', endPre - startPre)
                    loss += (np.power((R - prediction), 2) + regularization)

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
    '''
    # matrix test
    R = [
        [5, 3, 0, 1],
        [4, 0, 3, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ]
    R = np.array(R)
    '''
    print('read pre data:', datetime.now())
    R_DF, R_ARR, R_testX, R_testY = data.dataSetPre()
    print('read pre data end:', datetime.now())

    P, Q = matrixFactorization(R_ARR, 100)  # 取R（M，N）中小的一位为k

    SGD(P, Q, R_ARR, R_DF, R_testX, R_testY)

    '''
    w = 0
    for row in P:
        for column in Q:
            print(np.dot(row, column.T))
            w += 1

    print(w)
    '''
