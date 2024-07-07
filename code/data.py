import pandas as pd
import numpy as np


import os
ROOT_DIR = ('C:\\Users\\USER\\.')
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10000

ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in
               open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])
ratings_df['UserID'] = ratings_df['UserID'].astype(np.uint32)
ratings_df['MovieID'] = ratings_df['MovieID'].astype(np.uint32)
ratings_df['Rating'] = ratings_df['Rating'].astype(np.uint32)
ratings_df['Timestamp'] = ratings_df['Timestamp'].astype(np.uint32)

movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

#영화 id를 영화 제목에 매핑
movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

#모든 값을 정수형으로
ratings_df = ratings_df.applymap(int)


#유저별로 시청한 영화를 저장할 딕셔너리 초기화
users_dict = {user : [] for user in set(ratings_df["UserID"])}

#시간순 정렬
ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)

#평점이 4점 이상인 경우 users_dict_for_history_len에 해당 영화와 평점 저장
ratings_df_gen = ratings_df.iterrows()
users_dict_for_history_len = {user : [] for user in set(ratings_df["UserID"])}
for data in ratings_df_gen:
    users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    if data[1]['Rating'] >= 4:
        users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))


#유저가 평점 4점 이상을 준 영화의 개수를 계산
users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["UserID"])]


#저장
np.save("./data/user_dict.npy", users_dict)
np.save("./data/users_histroy_len.npy", users_history_lens)