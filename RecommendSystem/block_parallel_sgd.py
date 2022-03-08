from datetime import datetime
from multiprocessing import Pool
import numpy as np
from XieZhiyao_Assignment2 import dataPre as data

threshold = 0.0001  # final loss
Lamtha = 0.02  # Normolization factor
Gamma = 0.01  # learn rate
epochs = 1500  # number of iteration
k_embedding_factor = 100
thread_num = 4


def divBlock_matrix(matrix):
    row_list = []
    column_list = []
    row_len = int(matrix.shape[0] / thread_num)
    column_len = int(matrix.shape[1] / thread_num)
    row_rest = matrix.shape[0] % thread_num
    column_rest = matrix.shape[1] % thread_num

    for thread_row_index in range(thread_num):
        for thread_column_index in range(thread_num):

            row_list.append(thread_row_index * row_len)
            column_list.append(thread_column_index * column_len)
            row_list.append((thread_row_index + 1) * row_len)
            column_list.append((thread_column_index + 1) * column_len)

            if thread_column_index == (thread_num - 1):
                column_list[-1] += column_rest
            if thread_row_index == (thread_num - 1):
                row_list[-1] += row_rest

    return row_list, column_list


def divBlock_sequence(block_num, thread_num):
    # block_num = [1, 2, ... 15, 16]
    block = []
    for i in range(0, thread_num):  # 0~3
        for j in range(thread_num):  # 0~3
            block.append(block_num[j * thread_num + (i + j) % thread_num])
    # print('block: ', block)
    return block


def matrixFactorization(R, k):
    P = np.random.normal(-.05, .05, (R.shape[0], k))
    Q = np.random.normal(-.05, .05, (R.shape[1], k))

    return P, Q


def initial_value(R_hat):
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

    return Mu, R_count, b1, b2


# test RMSE
def RMSE(Mu, b1, b2):
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
        pre = np.dot(P[userID_Test_index], Q[movieID_Test_index]) + b1[userID_Test_index] + b2[movieID_Test_index] + Mu
        RMSE += np.power(rating_Test - pre, 2)
        # print(f'rating:{rating_Test},predict_ratings:{pre},user_id:{userID_Test},movie_id:{movieID_Test},uidindex:{userID_Test_index},midindex:{movieID_Test_index}')

    RMSE = (RMSE / R_testX.shape[0]) ** 0.5
    return RMSE


# !!! 有空再来写一下这个东西
# def share_memory_for_ProcessCommunication():

"""
    
    (update_parameter: & multiProcess: & parellel_sgd:)
    
    Used for test the multiProcess for single R in matrix

"""


def update_parameter(i, j, P, Q, b1, b2, Mu, R):
    # print("SubProcess>>> pid={}".format(os.getpid()))

    '''
    dot = 0
    for k in range(k_embedding_factor):
        dot += P[i, k] * Q[j, k]
    '''
    dot = np.dot(P[i], Q[j])
    prediction = dot + b1[i] + b2[j] + Mu
    error = R - prediction
    # print('updata para:')
    b1[i] = b1[i] + Gamma * (error - Lamtha * b1[i])
    b2[j] = b2[j] + Gamma * (error - Lamtha * b2[j])

    '''
    P_Norm_dot = 0
    Q_Norm_dot = 0
    # print(f'开始梯度下降：,pid:{os.getpid()}')
    for k in range(k_embedding_factor):
        P[i, k] = P[i, k] + Gamma * (error * Q[j, k] - Lamtha * P[i, k])
        Q[j, k] = Q[j, k] + Gamma * (error * P[i, k] - Lamtha * Q[j, k])
        # print('updata reg:')
        P_Norm_dot += np.linalg.norm(P[i, k])
        Q_Norm_dot += np.linalg.norm(Q[j, k])
    

    P_Norm = np.power(P_Norm_dot, 2)
    Q_Norm = np.power(Q_Norm_dot, 2)
    '''
    P[i] = P[i] + Gamma * (error * Q[j] - Lamtha * P[i])
    Q[j] = Q[j] + Gamma * (error * P[i] - Lamtha * Q[j])

    # print('updata reg:')
    # P_Norm = np.linalg.norm(P[i])
    P_Norm = np.power(np.linalg.norm(P[i]), 2)
    # Q_Norm = np.linalg.norm(Q[j])
    Q_Norm = np.power(np.linalg.norm(Q[j]), 2)
    regularization = Lamtha * (np.power(b1[i], 2) + np.power(b2[j], 2) + P_Norm + Q_Norm)

    loss = (np.power((R - prediction), 2) + regularization)
    # print(f'梯度：{loss},pid:{os.getpid()}')

    return loss


def multiProcess(pool, i, j, P, Q, b1, b2, Mu, R):
    # loss = []
    dots = []
    dot = pool.apply_async(update_parameter, args=(i, j, P, Q, b1, b2, Mu, R))

    '''
    dots.append(dot)
    for dot in dots:
        loss.append(dot.get())
    '''

    return dot


# single parallel sgd
def parellel_sgd(P, Q, R_hat, R_DF, R_testX, R_testY):
    now = datetime.now()
    print('sgd start:', now)

    Mu, R_count, b1, b2 = initial_value(R_hat)

    for epoch in range(epochs):

        start = datetime.now()
        loss = 0  # initial loss
        pool = Pool(thread_num)
        dots = []
        subloss = []

        for i, row in enumerate(R_hat):
            for j, R in enumerate(row):

                if not np.isnan(R):
                    # print('prediction:')
                    startMulti = datetime.now()
                    dot = multiProcess(pool, i, j, P, Q, b1, b2, Mu, R)
                    sizes = []
                    # for _ in [P, Q, R]:
                    #     sizes.append(_.nbytes)
                    # print('进程传递数据量：',sum(sizes)/1024/1024/8, 'MB')
                    # pipe_speed = '10000MB'
                    # print('进程传递速度:',sum(sizes)/1024/1024/8/1000,'s')
                    endMulti = datetime.now()
                    dots.append(dot)
                    # print('one_MultiProcess time cost:', endMulti - startMulti)
        count = 0
        for dot in dots:
            startDot = datetime.now()
            print(type(dot))
            # subloss.append(dot.get())
            endDot = datetime.now()
            print("dot time cost: ", endDot - startDot)
        # print('sum subloss:', np.array(subloss, dtype=float).sum())

        loss = sum(subloss) / R_count

        if loss <= threshold:
            break

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
            pre = np.dot(P[userID_Test_index], Q[movieID_Test_index]) + b1[userID_Test_index] + b2[movieID_Test_index] + Mu
            RMSE += np.power(rating_Test - pre, 2)
            # print(f'rating:{rating_Test},predict_ratings:{pre},user_id:{userID_Test},movie_id:{movieID_Test},uidindex:{userID_Test_index},midindex:{movieID_Test_index}')

        RMSE = (RMSE / R_testX.shape[0]) ** 0.5

        end = datetime.now()
        print("迭代次数:", (epoch + 1), "loss:", loss, "RMSE:", RMSE, "Time:", end - start)


"""
    
    (update_parameter_block: & block_parallel_sgd:)
    
    Used for test the multiProcess for div block R in matrix
    
"""


def update_parameter_block(row_start, column_start, row_end, column_end, P, Q, R, b1, b2, Mu):
    # loop from P[row_start, column_start] to P[row_end, column_end]
    start_block = datetime.now()
    loss = 0
    for row in range(row_start, row_end):
        for column in range(column_start, column_end):
            if not np.isnan(R[row, column]):
                dot = np.dot(P[row], Q[column])
                prediction = dot + b1[row] + b2[column] + Mu
                error = R[row, column] - prediction

                # print('updata para:')
                b1[row] = b1[row] + Gamma * (error - Lamtha * b1[row])
                b2[column] = b2[column] + Gamma * (error - Lamtha * b2[column])
                P[row] = P[row] + Gamma * (error * Q[column] - Lamtha * P[row])
                Q[column] = Q[column] + Gamma * (error * P[row] - Lamtha * Q[column])

                # print('updata reg:')
                # P_Norm = np.linalg.norm(P[i])
                P_Norm = np.power(np.linalg.norm(P[row]), 2)
                # Q_Norm = np.linalg.norm(Q[j])
                Q_Norm = np.power(np.linalg.norm(Q[column]), 2)
                regularization = Lamtha * (np.power(b1[row], 2) + np.power(b2[column], 2) + P_Norm + Q_Norm)

                loss += (np.power((R[row, column] - prediction), 2) + regularization)
    end_block = datetime.now()
    print('one block time cost:', end_block - start_block)
    return loss, P, Q, b1, b2


# block parallel sgd
def block_parallel_sgd(R_ARR, row_list, column_list, P, Q):
    Mu, R_count, b1, b2 = initial_value(R_ARR)
    block_num = [i for i in range(thread_num * thread_num)]
    block_seq = divBlock_sequence(block_num, thread_num)
    # print('block_num:', seq)
    # block 4 * 4 :[0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14]

    for epoch in range(epochs):
        loss = 0
        block_loss = []
        pool = Pool(thread_num)
        start = datetime.now()

        for index in range(0, len(block_seq), thread_num):
            # print('start index:', row_list[index], column_list[index])
            # print('end index:', row_list[index + 1], column_list[index + 1])

            # print('start index:', row_list[block_seq[index] * 2], column_list[block_seq[index] * 2])
            # print('end index:', row_list[block_seq[index] * 2 + 1], column_list[block_seq[index] * 2 + 1])
            # row_start = row_list[block_seq[index] * 2]
            # column_start = column_list[block_seq[index] * 2]
            # row_end = row_list[block_seq[index] * 2 + 1]
            # column_end = column_list[block_seq[index] * 2 + 1]
            block_dots = []

            '''
                ！！！ 再查一次边界：610/9724（609/9723） 的取值为什么没有溢出，解决：生成坐标时减1 or 计算划分均值时取上界还是下界的问题
            '''

            start_groupMulti = datetime.now()
            for i in range(thread_num):
                block_dot = pool.apply_async(update_parameter_block, args=(
                    row_list[block_seq[index + i] * 2], column_list[block_seq[index + i] * 2],
                    row_list[block_seq[index + i] * 2 + 1], column_list[block_seq[index + i] * 2 + 1],
                    P, Q, R_ARR, b1, b2, Mu))
                block_dots.append(block_dot)

            end_groupMulti = datetime.now()
            print('finish one group Multi, time cost:', end_groupMulti - start_groupMulti)
            start_get = datetime.now()
            for block_dot in block_dots:
                # print('loss return:', block_dot.get())
                loss, P, Q, b1, b2 = block_dot.get()
                block_loss.append(loss)
            end_get = datetime.now()
            print('.get() time cost:', end_get - start_get)

        loss = np.array(block_loss).sum() / R_count
        end = datetime.now()

        rmse = RMSE(Mu, b1, b2)
        print("迭代次数:", (epoch + 1), "loss:", loss, "Time cost:", end - start, "RMSE:", rmse)


if __name__ == '__main__':
    print('read pre data:', datetime.now())
    R_DF, R_ARR, R_testX, R_testY = data.dataSetPre()
    print('read pre data end:', datetime.now())
    P, Q = matrixFactorization(R_ARR, k_embedding_factor)

    # parellel_sgd(P, Q, R_ARR, R_DF, R_testX, R_testY)
    row_list, column_list = divBlock_matrix(R_ARR)

    block_parallel_sgd(R_ARR, row_list, column_list, P, Q)
