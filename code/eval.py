#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from env import OfflineEnv
from recommendation import DRRAgent

import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

ROOT_DIR = ('C:\\Users\\USER\\.')
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')
STATE_SIZE = 10

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


if __name__ == "__main__":
    print('Data loading...')

    # Loading datasets
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

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load('.\\data\\user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load('.\\data\\users_histroy_len.npy')

users_num = max(ratings_df["UserID"]) + 1
items_num = max(ratings_df["MovieID"]) + 1

#Evaluation Setting

eval_users_num = int(users_num * 0.2)
eval_items_num = items_num
eval_users_dict = {k: users_dict.item().get(k) for k in range(users_num - eval_users_num, users_num)}
eval_users_history_lens = users_history_lens[-eval_users_num:]

print(len(eval_users_dict), len(eval_users_history_lens))


def evaluate(recommender, env, check_movies=False, top_k=False):

    episode_reward = 0
    steps = 0
    mean_precision = 0
    mean_ndcg = 0
    # 환경 리셋
    user_id, items_ids, done = env.reset()

    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')
        print('history items : \n', np.array(env.get_items_names(items_ids)))

    while not done:

        # Observe current state & Find action
        # Embedding 해주기
        user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
        # SRM으로 state 출력
        state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        # Action(ranking score) 출력
        action = recommender.actor.network(state)
        # Item 추천
        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)
        if check_movies:
            print(f'recommended items ids : {recommended_item}')
            print(f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')


        next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)
        if top_k:
            correct_list = [1 if r > 0 else 0 for r in reward]
            # ndcg
            dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])
            mean_ndcg += dcg / idcg

            # precision
            correct_num = top_k - correct_list.count(0)
            mean_precision += correct_num / top_k

        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

        if check_movies:
            print(
                f'precision : {correct_num / top_k}, dcg : {dcg:0.3f}, idcg : {idcg:0.3f}, ndcg : {dcg / idcg:0.3f}, reward : {reward}')
            print()
        break

    if check_movies:
        print(
            f'precision : {round(correct_num / top_k, 4)}, dcg : {round(dcg, 4):0.4f}, idcg : {round(idcg, 4):0.4f}, ndcg : {round(dcg / idcg, 4):0.4f}, reward : {reward}')
        print()

    if check_movies:
         print(f'precision : {mean_precision / steps}, ngcg : {mean_ndcg / steps}, episode_reward : {episode_reward}')
         print()

    return mean_precision / steps, mean_ndcg / steps, episode_reward


def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r) / np.log2(i + 2)
        idcg += (ir) / np.log2(i + 2)
    return dcg, idcg



saved_actor = '.trail-2024-06-23-13\\actor_10000_fixed.h5'
saved_critic = '.trail-2024-06-23-13\\critic_10000_fixed.h5'

tf.keras.backend.set_floatx('float32')

TOP_K = 5

#initialize
sum_precision = 0
sum_ndcg = 0
total_reward = 0

end_evaluation = len(eval_users_dict) - 1

for i, user_id in enumerate(eval_users_dict.keys()):
    env = OfflineEnv(eval_users_dict, eval_users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.load_model(saved_actor, saved_critic)
    precision, ndcg, episode_reward = evaluate(recommender, env, check_movies=True, top_k=TOP_K)  # 변경된 부분
    sum_precision += precision
    sum_ndcg += ndcg
    total_reward += episode_reward  #누적 보상값

    if i > end_evaluation:
        break
print(f'precision@{TOP_K} : {round(sum_precision / len(eval_users_dict), 4)}, ndcg@{TOP_K} : {round(sum_ndcg / len(eval_users_dict), 4)}, total reward: {total_reward}')

