import numpy as np


class OfflineEnv(object):

    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict #유저별 시청 영화&평점 정보 담은 딕셔너리
        self.users_history_lens = users_history_lens #각 유저의 평점 기록 길이
        self.items_id_to_name = movies_id_to_movies #영화 id와 영화 제목을 매핑

        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        #fixed_user_id가 주어지면 해당 값 사용하고, 그렇지 않으면 랜덤하게 선택된 사용자
        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        
        #현재 유저가 본 영화와 평점 정보
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        
        #현재 상태에서의 영화 리스트
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        
        #추천된 아이템 집합
        self.recommended_items = set(self.items)
        
        #추천 가능한 최대 아이템 개수
        self.done_count = 3000

    def _generate_available_users(self):
        #state size보다 많은 영화를 본 유저 필터링
        #state_size보다 많은 영화를 본 경우에 avaliable_users에 추가
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users

    def reset(self):
        #초기화
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done

    def step(self, action, top_k=False):
        #여기서 action은 actor network의 output인 action이 아니라 추천된 아이템을 나타냄
        reward=0
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                # act가 user_items에 있고, 추천된 적이 없으며, 평점이 4 이상인 경우
                if act in self.user_items.keys() and act not in self.recommended_items and self.user_items[act] > 3:
                    correctly_recommended.append(act)
                    rewards.append(1)  # 보상을 1로 설정
                else:
                    rewards.append(0)  # 그렇지 않은 경우 보상을 0으로 설정
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            # 단일 아이템에 대한 처리
            reward = 0  # 기본 보상을 0으로 설정
            if action in self.user_items.keys() and action not in self.recommended_items and self.user_items[action] > 3:
                reward = 1  # 조건을 만족하면 보상을 1로 설정
            self.recommended_items.add(action)
            if reward > 0:
                self.items = self.items[1:] + [action]

        # 추천할 수 있는 아이템의 수가 제한에 도달하거나 모든 아이템이 추천되었을 때 done 상태로 설정
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= len(self.user_items):
            self.done = True

        return self.items, reward, self.done, {}

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names