import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from datetime import datetime
from replay_buffer import PriorityExperienceReplay
from actor_network import Actor
from critic_network import Critic
from embedding import UserMovieEmbedding
from state_representation import DRRAveStateRepresentation

class DRRAgent:
    def __init__(self, env, users_num, items_num, state_size, is_test=False, use_wandb=False):
        self.env = env
        self.users_num = users_num
        self.items_num = items_num

        # Initialize dimensions and hyperparameters before loading embeddings
        self.embedding_dim = 100
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001
        self.replay_memory_size = 1000000
        self.batch_size = 32
        self.num_quantiles = 20

        #actor, critic 네트워크 초기화
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau, self.num_quantiles)
        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        #embedding network 초기화
        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)), np.zeros((1,))])

        self.save_model_weight_dir = f".\\trail-{datetime.now().strftime('%Y-%m-%d-%H')}"
        images_dir = os.path.join(self.save_model_weight_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        embedding_save_file_dir = '.\\final_weight.h5'
        assert os.path.exists(
            embedding_save_file_dir), f"embedding save file directory: '{embedding_save_file_dir}' is wrong."
        self.embedding_network.load_weights(embedding_save_file_dir)

        #state representation module 네트워크 초기화
        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)), np.zeros((1, state_size, 100))])

        # priority experience replay
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6


        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1) / 1000000
        #노이즈 추가
        self.std = 1.0

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr",
                       entity="mj",
                       config={'users_num': users_num,
                               'items_num': items_num,
                               'state_size': state_size,
                               'embedding_dim': self.embedding_dim,
                               'actor_hidden_dim': self.actor_hidden_dim,
                               'actor_learning_rate': self.actor_learning_rate,
                               'critic_hidden_dim': self.critic_hidden_dim,
                               'critic_learning_rate': self.critic_learning_rate,
                               'discount_factor': self.discount_factor,
                               'tau': self.tau,
                               'replay_memory_size': self.replay_memory_size,
                               'batch_size': self.batch_size,
                               'std_for_exploration': self.std})

    def calculate_td_target(self, rewards, next_quantiles, dones):
        rewards = np.array(rewards)[:, None]  # 배열로 변환하는 코드
        dones = np.array(dones)[:, None]
        #rewards: 현재 상태에서의 보상, 1-dones: 에피소드가 종료되지 않을 경우 1, 종료되면 0, next_quantiles: critic network의 타겟 네트워크에서 예측된 값
        td_targets = rewards + self.discount_factor * (1 - dones) * next_quantiles
        # print("TD Targets:", td_targets)
        return td_targets
    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids is None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))

        #모든 아이템의 임베딩 벡터를 가져옴
        items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)
        #action: actor network에서 생성된 continuous parameter vector
        action = tf.transpose(action, perm=(1, 0))
        #top k가 true면 상위 k개의 item 추천
        #item embedding과 action 벡터의 dot product 연산을 통해 ranking score 계산
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1, 0)))[0][-top_k:]
            return items_ids[item_indice]
        #top k가 false면 가장 높은 점수를 받은 아이템 추천
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]


    def train(self, max_episode_num, top_k=False, load_model=False):
        self.actor.update_target_network()
        self.critic.update_target_network()


        #각 에피소드별 precision, q loss, reward 저장하기 위한 리스트
        episodic_precision_history = []
        episodic_q_loss_history = []
        episodic_reward_history = []


        for episode in range(max_episode_num):
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            total_q_loss = 0

            mean_action = 0
            user_id, items_ids, done = self.env.reset()

            while not done:
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                action = self.actor.network(state)
                
                #self.epsilon: 초기에 1.0으로 설정되어있기 때문에 무작위로 생성된 난수가 1.0보다 대부분 작음, 따라서 탐험을 자주 하게 됨
                #not self.is_test 인 경우에 exploration을 하지 않고 self.is_test인 경우에는 탐험
                if self.epsilon > np.random.uniform() and not self.is_test:
                    #self.epsilon값을 self.epsilon_decay만큼 감소시켜서 시간이 지남에 따라 exploration 빈도가 줄어들고 exploitation이 늘어나게 됨
                    self.epsilon -= self.epsilon_decay
                    # noise를 추가
                    action += np.random.normal(0, self.std, size=action.shape)

                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)

                if isinstance(reward, list):
                    reward = sum(reward)

                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                self.buffer.append(state, action, reward, next_state, done)
                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(
                        self.batch_size)

                    #next state에 대한 action
                    target_next_action = self.actor.target_network(batch_next_states)
                    #분포 예측
                    target_next_quantiles = self.critic.target_network([batch_next_states, target_next_action])
                    #td 계산
                    td_targets = self.calculate_td_target(batch_rewards, target_next_quantiles, batch_dones)

                    q_loss += self.critic.train(inputs=[batch_states, batch_actions], td_targets=td_targets,
                                                weight_batch=weight_batch)

                    #replay buffer per
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    q_loss += self.critic.train(inputs=[batch_states, batch_actions], td_targets=td_targets,
                                                weight_batch=weight_batch)

                    total_q_loss += q_loss
                    s_grads = self.critic.dq_da([batch_actions, batch_states])
                    self.actor.train(batch_states, s_grads)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward

                mean_action += np.sum(action[0]) / (len(action[0]))
                steps += 1

                if reward > 0:
                    correct_count += 1

                print(
                    f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}',
                    end='\r')

                if done:
                    print()
                    precision = int(correct_count / steps * 100)
                    episodic_precision_history.append(precision)
                    average_q_loss = total_q_loss / steps if steps > 0 else 0
                    episodic_q_loss_history.append(average_q_loss)
                    episodic_reward_history.append(episode_reward)

                    print(
                        f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss / steps}, mean_action : {mean_action / steps}')
                    if self.use_wandb:
                        wandb.log({'precision': precision, 'total_reward': episode_reward, 'epsilone': self.epsilon,
                                   'q_loss': average_q_loss, 'mean_action': mean_action / steps})

            #시각화
            if (episode + 1) % 10000 == 0 or episode == max_episode_num - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(episodic_q_loss_history)
                plt.title(f'Q-Loss History up to Episode {episode + 1}')
                plt.xlabel('Episode')
                plt.ylabel('Q-Loss')
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images\\q_loss_{episode + 1}.png'))
                plt.close()

                plt.figure(figsize=(10, 5))
                plt.plot(episodic_reward_history)
                plt.title(f'Total Reward History up to Episode {episode + 1}')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images\\total_reward_{episode + 1}.png'))
                plt.close()

                plt.plot(episodic_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images\\training_precision_%_top_5.png'))

            if (episode + 1) % 10000 == 0:
                save_path = os.path.join(self.save_model_weight_dir, f'user_movie_embedding_{episode + 1}.h5')
                self.embedding_network.save_weights(save_path)
                print(f"Saved embedding weights at episode {episode + 1} to {save_path}")

            if (episode + 1) % 10000 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images\\training_precision_%_top_5.png'))

            if (episode + 1) % 10000 == 0 or episode == max_episode_num - 1:
                self.save_model(os.path.join(self.save_model_weight_dir, f'actor_{episode + 1}_fixed.h5'),
                                os.path.join(self.save_model_weight_dir, f'critic_{episode + 1}_fixed.h5'))

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
