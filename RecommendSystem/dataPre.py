import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split


def datasetDiv(data, labels):
    # data:PQ  labels:R
    # csv中取出一部分数据（重复用户or重复电影）来，不参与构建稀疏矩阵，但要保证每一个user和item都有参与训练
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=None)

    return trainX, testX, trainY, testY


def dataSetPre():
    # dataSet: movieslen

    df_movie = pd.read_csv("./ml-latest-small/movies.csv",
                           sep=",", skiprows=1, engine="python",
                           names=['MovieID', 'Title', 'Genres'])

    df_rating = pd.read_csv("./ml-latest-small/ratings.csv",
                            sep=",", skiprows=1, engine="python",
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    # 计算电影中每个题材的次数
    genre_count = collections.defaultdict(int)
    for genres in df_movie["Genres"].str.split("|"):
        for genre in genres:
            genre_count[genre] += 1

    print('genre_count:', len(genre_count))

    # 只保留最有代表性的题材
    def get_highrate_genre(x):
        sub_values = {}
        for genre in x.split("|"):
            sub_values[genre] = genre_count[genre]
        return sorted(sub_values.items(), key=lambda x: x[1], reverse=True)[0][0]

    df_movie["Genres"] = df_movie["Genres"].map(get_highrate_genre)
    # print('df_movie:', df_movie)

    df_rating = pd.DataFrame(df_rating, dtype='object')
    df_movie = pd.DataFrame(df_movie, dtype='object')
    # print('df_movie:', df_movie)
    # print('df_rating:', df_rating)

    df_rating.to_csv("./generate_data/newRating.csv", index=False)
    df_movie.to_csv("./generate_data/newMovie.csv", index=False)

    # print("df_rating[['UserID', 'MovieID']]:", df_rating[['UserID', 'MovieID']])
    R_trainX, R_testX, R_trainY, R_testY = datasetDiv(df_rating[['UserID', 'MovieID']], df_rating[['Rating']])
    R_trainX.to_csv("./generate_data/trainSetX.csv", index=False)
    R_trainY.to_csv("./generate_data/trainSetY.csv", index=False)
    R_testX.to_csv("./generate_data/testSetX.csv", index=False)
    R_testY.to_csv("./generate_data/testSetY.csv", index=False)

    # test shuffle
    '''
    shuffle = []
    shuffle = pd.read_csv("./trainSetX.csv", sep=",")
    count = np.unique(shuffle['UserID'])
    print("count:", len(count))
    '''

    # print('df_rating_userID:')
    row = np.unique(df_rating['UserID'])
    # print('df_rating_movieID:')
    # column = max(df_rating['MovieID'])
    column = np.unique(df_rating['MovieID'])
    print('row:', len(row), 'column:', len(column))

    # row_index: 0 ~ 609; column_index: 0 ~ 193608
    row_name = []
    for i in row:
        row_name.append("User" + str(i))
    # print('row_name:', row_name)
    column_name = []
    for i in column:
        column_name.append("Movie" + str(i))
    # print('column_name:', column_name)

    matrixR = pd.DataFrame(columns=column_name, index=row_name)

    """
    shape: 610 * 9724 
       i1 i2 i3 ... im
    u1
    u2
    u3
    .
    .
    .
    un
    """
    print('test:')
    for i in range(R_trainX['UserID'].shape[0]):
        # print('UserID:', R_trainX.iloc[i, 1], 'MovieID:', R_trainX.iloc[i, 2], 'Rating:', R_trainY.iloc[i, 1])
        # -1解决索引边界问题
        # matrixR.iloc[(R_trainX.iloc[i, 1] - 1), (R_trainX.iloc[i, 2] - 1)] = R_trainY.iloc[i, 1]
        # print(str(R_trainX.iloc[i, 0]),str(R_trainX.iloc[i, 1]),R_trainY.iloc[i, 0])
        matrixR.loc["User" + str(R_trainX.iloc[i, 0]), "Movie" + str(R_trainX.iloc[i, 1])] = R_trainY.iloc[i, 0]
    # print(matrixR.dtypes)
    # print('matrixR:', matrixR)
    matrixR_array = np.array(matrixR, dtype=float)

    # print(matrixR.dtype)

    return matrixR, matrixR_array, R_testX, R_testY
