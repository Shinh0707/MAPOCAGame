from termcolor import colored
from tabulate import tabulate
import pygame
import math
import datetime
import glob
import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAgentAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None= None):
        # x shape: (batch_size, player_num, seq_len, feature_dim)
        batch_size, player_num, seq_len, feature_dim = x.shape

        # Reshape to (seq_len, batch_size * player_num, feature_dim)
        x_reshaped = x.transpose(1, 2).transpose(0, 1).reshape(
            seq_len, batch_size * player_num, feature_dim)

        if mask is not None:
            # mask shape: (batch_size, player_num, seq_len)
            # Reshape to (batch_size * player_num, seq_len)
            mask_reshaped = mask.reshape(batch_size * player_num, seq_len)
        else:
            mask_reshaped = None

        # Apply attention
        attended, _ = self.attention.forward(
            x_reshaped, x_reshaped, x_reshaped, key_padding_mask=mask_reshaped)

        # Reshape back to (batch_size, player_num, seq_len, feature_dim)
        attended = attended.reshape(
            seq_len, batch_size, player_num, feature_dim).transpose(0, 1).transpose(1, 2)

        return attended

class AgentNetwork(nn.Module):
    def __init__(self, observation_dim=7, state_dim=7, team_state_dim=5, action_dim=4, embed_dim=64, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.team_state_dim = team_state_dim

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.observation_embedding = nn.Linear(observation_dim, embed_dim)
        self.team_state_embedding = nn.Linear(team_state_dim, embed_dim)

        self.observation_attention = MultiAgentAttention(
            embed_dim, num_heads)
        self.team_attention = MultiAgentAttention(
            embed_dim, num_heads)

        self.fc_shared = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),  # 状態、観測、チーム状態の3つの埋め込みを結合
            nn.ReLU()
        )

        self.fc_policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.fc_value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

        # Attention層の特別な初期化
        for attention in [self.observation_attention, self.team_attention]:
            nn.init.xavier_uniform_(attention.attention.in_proj_weight)
            nn.init.constant_(attention.attention.in_proj_bias, 0)
            nn.init.constant_(attention.attention.out_proj.bias, 0)
            nn.init.xavier_uniform_(attention.attention.out_proj.weight)

        # Policy headの最後の層を特別に初期化
        nn.init.orthogonal_(self.fc_policy[-1].weight, gain=0.01)
        nn.init.constant_(self.fc_policy[-1].bias, 0)

        # Value headの最後の層を特別に初期化
        nn.init.orthogonal_(self.fc_value[-1].weight, gain=1.0)
        nn.init.constant_(self.fc_value[-1].bias, 0)

    def forward(self, observations: torch.Tensor, obs_mask: torch.Tensor | None, team_states: torch.Tensor, team_mask: torch.Tensor | None):
        # 状態情報の処理
        state = observations[:, :, 0, :self.state_dim]

        state_embedded = self.state_embedding.forward(state)

        # 観測データの処理
        obs = observations[:, :, 1:, :]
        obs_mask = obs_mask[:,:,1:]
        obs_embedded = self.observation_embedding.forward(obs)

        # チーム状態の処理
        team_embedded = self.team_state_embedding.forward(team_states)

        # Attention処理
        obs_attended = self.observation_attention.forward(
            obs_embedded,
            mask=obs_mask)

        team_attended = self.team_attention.forward(
            team_embedded,
            mask=team_mask)

        # Global average pooling
        obs_attended_pooled = obs_attended.mean(
            dim=2)  # (batch_size, player_num, embed_dim)
        team_attended_pooled = team_attended.mean(
            dim=2)  # (batch_size, player_num, embed_dim)

        # 状態、注意を適用した観測、チーム状態を結合
        combined = torch.cat(
            [state_embedded, obs_attended_pooled, team_attended_pooled], dim=-1)

        # 共有層
        shared_features = self.fc_shared.forward(combined)

        # Policy (行動確率) の計算
        action_logits = self.fc_policy.forward(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Value (状態価値) の計算
        value: torch.Tensor = self.fc_value.forward(shared_features).squeeze(-1)

        return action_probs, value

def create_mask(data: list, max_num: int):
    if len(data) > max_num:
        data = data[:max_num]
    if len(data) == max_num:
        return torch.tensor(data, dtype=torch.float32), torch.zeros(max_num, dtype=torch.bool)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    data_tensor = F.pad(data_tensor, (0, 0, 0, max_num - len(data)))
    mask = torch.ones(max_num, dtype=torch.bool)
    mask[:len(data)] = False
    return data_tensor, mask

class GameEnvironment:
    MAX_HP = 100
    MAX_AMMO = 10
    AMMO_DAMAGE = 20
    DIRECTIONS = 8
    TO_RAD = 2 * np.pi / DIRECTIONS
    RELOAD_AMMO_PER_FRAME = 0.1


    def __init__(self, num_teams, players_per_team, map_size):
        self.num_teams = num_teams
        self.players_per_team = players_per_team
        self.total_players = num_teams * players_per_team
        self.max_num_obs = self.total_players
        self.map_size = np.array(map_size)
        self.max_map_size = np.linalg.norm(map_size, ord=2)
        self.action_space = 5
        self.max_time = 1000
        self.team_colors = [
            'red', 'blue', 'green', 'yellow', 'magenta', 'cyan'
        ]  # チームの色をリストで定義
        self.player_stats = []
        self.reset()

    def reset(self):
        self.players = []
        self.player_stats = []  # プレイヤーの戦績をリセット
        id = 0
        for team in range(self.num_teams):
            for _ in range(self.players_per_team):
                self.players.append({
                    'position': np.random.rand(2) * self.map_size,
                    'direction': np.random.randint(0, GameEnvironment.DIRECTIONS-1),
                    'hp': GameEnvironment.MAX_HP,
                    'ammo': GameEnvironment.MAX_AMMO,
                    'team': team,
                    'id' : id
                })
                self.player_stats.append({
                    'team': team,
                    'kills': 0,
                    'damage_dealt': 0,
                    'team_attack': 0,
                    'team_kill': 0,
                    'shots_fired': 0,
                    'shots_hitted': 0,
                    'survived': True
                })
                id += 1
        self.bullets = []
        self.time = 0
        return self.get_state()
    
    def calculate_rewards(self, actions):
        rewards = torch.zeros(self.total_players)

        for i, (player, action) in enumerate(zip(self.players, actions)):
            if player['hp'] <= 0:
                continue

            # 1. 生存報酬
            rewards[i] += 0.1  # 各ステップで生存していることに対する小さな報酬

            # 2. 弾薬管理報酬
            if player['ammo'] < GameEnvironment.MAX_AMMO * 0.2:
                rewards[i] -= 0.05  # 弾薬が少ない場合のペナルティ

            # 3. 移動報酬
            if action == 2:  # 移動アクション
                rewards[i] += 0.01  # 移動に対する小さな報酬

            # 4. 射撃精度報酬
            if action == 3 and player['ammo'] >= 1:  # 射撃アクション
                rewards[i] -= 0.02  # 射撃に対する小さなペナルティ（命中時に相殺される）

            # 5. チームワーク報酬（後で計算）

            # 6. 戦略的位置取り報酬（マップ中央への接近）
            center = self.map_size / 2
            distance_to_center = np.linalg.norm(player['position'] - center)
            rewards[i] += 0.05 * (1 - distance_to_center /
                                  np.linalg.norm(self.map_size))

        return rewards

    def step(self, actions):
        self.time += 1
        rewards = self.calculate_rewards(actions)

        # Process actions
        for i, action in enumerate(actions):
            player = self.players[i]
            if player['hp'] > 0:
                if action == 0:  # Rotate clockwise
                    player['direction'] = (
                        player['direction'] + 1) % GameEnvironment.DIRECTIONS
                elif action == 1:  # Rotate counterclockwise
                    player['direction'] = (
                        player['direction'] - 1) % GameEnvironment.DIRECTIONS
                elif action == 2:  # Move forward
                    player['position'] += np.array(
                        [np.cos(player['direction']*GameEnvironment.TO_RAD), np.sin(player['direction']*GameEnvironment.TO_RAD)])
                    player['position'] = np.clip(
                        player['position'], 0, self.map_size - 1)
                elif action == 3 and player['ammo'] >= 1:  # Shoot
                    self.bullets.append({
                        'position': player['position'].copy(),
                        'direction': player['direction'],
                        'team': player['team'],
                        'shooter': player['id']
                    })
                    player['ammo'] -= 1
                    self.player_stats[i]['shots_fired'] += 1  # 発射数をカウント

        # Move bullets
        for bullet in self.bullets:
            bullet['position'] += np.array(
                [np.cos(bullet['direction']*GameEnvironment.TO_RAD), np.sin(bullet['direction']*GameEnvironment.TO_RAD)])

        # 衝突の処理
        collision_rewards = self._handle_collisions()
        rewards += collision_rewards

        # チームワーク報酬の計算
        teamwork_rewards = self._calculate_teamwork_rewards()
        rewards += teamwork_rewards

        # Regenerate ammo
        for player in self.players:
            player['ammo'] = min(
                player['ammo'] + GameEnvironment.RELOAD_AMMO_PER_FRAME, GameEnvironment.MAX_AMMO)

        return *self.get_state(), rewards, self.get_dones()

    def _handle_collisions(self):
        # Bullet-player collisions
        collision_rewards = torch.zeros(self.total_players)
        remove_bullets = []
        for i, bullet in enumerate(self.bullets):
            if i in remove_bullets:
                continue
            for j, player in enumerate(self.players):
                if bullet['shooter'] != player['id'] and player['hp'] > 0:
                    if np.linalg.norm(bullet['position'] - player['position']) < 0.5:
                        self.player_stats[bullet['shooter']
                                          ]['shots_hitted'] += 1
                        collision_rewards[bullet['shooter']] += 0.5  # 命中報酬
                        collision_rewards[j] -= 0.5 # 命中ペナルティ
                        old_hp = self.players[j]['hp']
                        self.players[j]['hp'] -= GameEnvironment.AMMO_DAMAGE
                        damage_dealt = old_hp - max(self.players[j]['hp'], 0)
                        self.player_stats[bullet['shooter']
                                          ]['damage_dealt'] += damage_dealt
                        if bullet['team'] != player['team']:
                            collision_rewards[bullet['shooter']
                                              ] += 1
                            if self.players[j]['hp'] <= 0:
                                # キル報酬
                                collision_rewards[bullet['shooter']] += 5
                        else:
                            self.player_stats[bullet['shooter']
                                              ]['team_attack'] += 1
                            # チームダメージのペナルティ
                            collision_rewards[bullet['shooter']] -= 2
                            if self.players[j]['hp'] <= 0:
                                self.player_stats[bullet['shooter']
                                                  ]['team_kill'] += 1
                                # TK penalty
                                collision_rewards[bullet['shooter']] -= 10
                        if self.players[j]['hp'] <= 0:
                            self.player_stats[bullet['shooter']
                                              ]['kills'] += 1
                            collision_rewards[j] -= 5
                            self.player_stats[j]['survived'] = False
                        remove_bullets.append(i)
                        break
            for j, bullet2 in enumerate(self.bullets[i+1:], i+1):
                if j in remove_bullets:
                    continue
                if np.linalg.norm(bullet['position'] - bullet2['position']) < 0.5:
                    collision_rewards[bullet['shooter']] += 0.1  # 弾丸衝突ボーナス
                    collision_rewards[bullet2['shooter']] += 0.1
                    remove_bullets.extend([i, j])
                    break
        
        self.bullets = [bullet for i, bullet in enumerate(self.bullets) if not i in remove_bullets]

        # Remove out-of-bounds bullets
        self.bullets = [b for b in self.bullets if 0 <= b['position']
                        [0] < self.map_size[0] and 0 <= b['position'][1] < self.map_size[1]]
        return collision_rewards
    
    def _calculate_teamwork_rewards(self):
        teamwork_rewards = torch.zeros(self.total_players)

        for i, player in enumerate(self.players):
            if player['hp'] <= 0:
                continue

            teammates = [p for j, p in enumerate(
                self.players) if p['team'] == player['team'] and j != i]

            # チームメイトとの距離に基づく報酬
            for teammate in teammates:
                distance = np.linalg.norm(
                    player['position'] - teammate['position'])
                if distance < 10:  # 近すぎる場合はペナルティ
                    teamwork_rewards[i] -= 0.02
                elif 10 <= distance <= 20:  # 適度な距離の場合は報酬
                    teamwork_rewards[i] += 0.05

            # チームメイトのカバー報酬
            for bullet in self.bullets:
                if bullet['team'] != player['team']:
                    for teammate in teammates:
                        if np.linalg.norm(bullet['position'] - teammate['position']) < 2 and np.linalg.norm(player['position'] - teammate['position']) < 5:
                            teamwork_rewards[i] += 0.1  # チームメイトをカバーしている報酬

        return teamwork_rewards

    def get_observations(self):
        global_state = self.get_global_state()
        observations = []
        masks = []
        for i, player in enumerate(self.players):
            obs = []
            if player['hp'] > 0:
                for j, other in enumerate(self.players):
                    if i != j and other['hp'] > 0:
                        relative_pos = other['position'] - player['position']
                        angle = np.arctan2(
                            relative_pos[1], relative_pos[0]) - player['direction'] * GameEnvironment.TO_RAD
                        if abs(angle) <= np.pi / 2:  # Within 180 degree field of view
                            distance = np.linalg.norm(relative_pos, ord=2)
                            # HP in 25% increments
                            hp_category = min(other['hp'] // 25, 3)
                            obs.append([
                                distance / self.max_map_size,
                                angle / np.pi,
                                hp_category / 3,
                                other['direction'] / GameEnvironment.DIRECTIONS,
                                float(other['team'] == player['team']),
                                1,
                                0
                            ])
                for bullet in self.bullets:
                    if bullet['shooter'] != i:
                        relative_pos = bullet['position'] - player['position']
                        angle = np.arctan2(
                            relative_pos[1], relative_pos[0]) - player['direction'] * GameEnvironment.TO_RAD
                        if abs(angle) <= np.pi / 2:  # Within 180 degree field of view
                            distance = np.linalg.norm(relative_pos, ord=2)
                            obs.append([
                                distance / self.max_map_size,
                                angle / np.pi,
                                0,
                                bullet['direction'] / GameEnvironment.DIRECTIONS,
                                float(other['team'] == player['team']),
                                0,
                                1
                            ])
            if len(obs) > 0:
                obs.sort(key=lambda d: d[0])
            if len(obs) == 0:
                obs.append([0.0]*7)
            obs.insert(0, global_state+self.get_player_state(player))
            padded_obs, mask = create_mask(obs, self.max_num_obs+1)
            observations.append(padded_obs)
            masks.append(mask)
        return torch.stack(observations).unsqueeze(0), torch.stack(masks).unsqueeze(0)

    def get_team_states(self):
        team_states = []
        team_masks = []
        for player in self.players:
            team_state = []
            for teammate in self.players:
                if teammate['team'] == player['team']:
                    if teammate['hp'] > 0:
                        team_state.append(self.get_team_player_state(teammate))
            if len(team_state) == 0:
                team_state = [[0.0]*6]
            padded_state, mask = create_mask(team_state, self.players_per_team)
            team_states.append(padded_state)
            team_masks.append(mask)
        return torch.stack(team_states).unsqueeze(0), torch.stack(team_masks).unsqueeze(0)
    
    def get_state(self):
        return self.get_observations(), self.get_team_states()
    
    def get_player_state(self, player):
        return [
            player['position'][0] / self.map_size[0],
            player['position'][1] / self.map_size[1],
            player['direction']/GameEnvironment.DIRECTIONS,
            player['hp'] / GameEnvironment.MAX_HP,
            player['ammo'] / GameEnvironment.MAX_AMMO
        ]
    
    def get_team_player_state(self, player):
        danger = 0.0
        if self.players_per_team > 1:
            danger = self.player_stats[player['id']
                                       ]['team_attack'] / (self.players_per_team-1)
        return [
            player['position'][0] / self.map_size[0],
            player['position'][1] / self.map_size[1],
            player['direction']/GameEnvironment.DIRECTIONS,
            player['hp'] / GameEnvironment.MAX_HP,
            player['ammo'] / GameEnvironment.MAX_AMMO,
            danger
        ]
    
    def get_global_state(self):
        return [
            self.time / self.max_time,  # 正規化された経過時間
            len(self.bullets) / (self.total_players * GameEnvironment.MAX_AMMO),  # 正規化された弾の総数
        ]
    
    def get_dims(self):
        """
        Returns Observation Dim, State Dim, Team State Dim, Action Dim
        """
        g_state_len = len(self.get_global_state())
        state_len = len(self.get_player_state(self.players[0]))
        team_state_len = len(self.get_team_player_state(self.players[0]))
        return g_state_len + state_len, g_state_len + state_len, team_state_len, self.action_space
    
    def is_done(self):
        if self.time >= self.max_time:  # Time limit
            return True
        alive_teams = set(player['team']
                          for player in self.players if player['hp'] > 0)
        return len(alive_teams) <= 1
    
    def get_dones(self):
        dones = torch.ones(self.total_players,dtype=torch.bool)
        if self.time < self.max_time:
            for i, player in enumerate(self.players):
                if player['hp'] > 0:
                    dones[i] = False
        return dones
    
    def get_player_stats(self):
        return self.player_stats

class MAPOCA:
    def __init__(self, env: GameEnvironment, network: AgentNetwork, lr=1e-4, gamma=0.99, lambda_=0.95, epsilon=0.1, value_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.num_epochs = 10  # 更新のエポック数
        self.clip_param = 0.2  # PPOのクリッピングパラメータ
        self.value_loss_coef = value_coef  # 価値損失の係数
        self.entropy_coef = entropy_coef  # エントロピー損失の係数
        self.max_grad_norm = 0.5  # 勾配クリッピングの最大ノルム
        self.MODEL_DIR = "mapoca_models"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.best_reward = float('-inf')
        self.best_performance = float('-inf')
        self.save_interval = 10  # エピソード数ごとにモデルを保存
        self.keep_best_n = 5  # 保持する上位nモデル数
        self.keep_interval = 100  # 保持する定期保存モデルの間隔

    def calc_performance_score(self, reward, policy_loss, value_loss, entropy):
        return reward - (policy_loss + value_loss) + entropy * 0.1

    def save_model(self, episode, reward, policy_loss, value_loss, entropy):
        # 性能スコアの計算（報酬、損失、エントロピーを考慮）
        performance_score = self.calc_performance_score(
            reward, policy_loss, value_loss, entropy)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapoca_model_ep{episode}_r{reward:.2f}_pl{policy_loss:.4f}_vl{value_loss:.4f}_e{entropy:.4f}_{timestamp}.pth"
        path = os.path.join(self.MODEL_DIR, filename)

        torch.save({
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': reward,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'performance_score': performance_score,
        }, path)

        print(f"Model saved to {path}")
        # 古いモデルを整理
        self.cleanup_old_models()

        return performance_score

    def cleanup_old_models(self):
        # モデルファイルのリストを取得
        model_files = glob.glob(os.path.join(
            self.MODEL_DIR, "mapoca_model_*.pth"))

        if len(model_files) <= self.keep_best_n:
            return  # モデルファイルが存在しない場合は何もしない

        # モデルを性能スコアでソート
        sorted_models = sorted(
            model_files,
            key=lambda x: torch.load(x)['performance_score'],
            reverse=True
        )

        # 上位n個のベストモデルを取得
        best_models = set(sorted_models[:self.keep_best_n])

        # 定期保存モデルを取得
        interval_models = set()
        for model in sorted_models:
            episode = int(model.split('_ep')[1].split('_')[0])
            if episode % self.keep_interval == 0:
                interval_models.add(model)

        # 削除するモデルを決定
        models_to_keep = best_models.union(interval_models)
        models_to_delete = set(model_files) - models_to_keep

        # 古いモデルを削除
        for model in models_to_delete:
            os.remove(model)
            print(f"Deleted old model: {model}")

    def load_best_model(self):
        model_files = glob.glob(os.path.join(
            self.MODEL_DIR, "mapoca_model_*.pth"))

        if not model_files:
            print("No saved models found.")
            return False

        best_model = None
        best_performance = float('-inf')

        for model_file in model_files:
            try:
                checkpoint = torch.load(model_file)
                performance = checkpoint['performance_score']

                if performance > best_performance:
                    best_performance = performance
                    best_model = model_file
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")

        if best_model:
            self.load_model(best_model)
            print(f"Loaded best model: {
                  best_model} with performance score: {best_performance}")
            return True
        else:
            print("No valid models found.")
            return False

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False

        try:
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episode = checkpoint['episode']
            reward = checkpoint['reward']
            policy_loss = checkpoint['policy_loss']
            value_loss = checkpoint['value_loss']
            entropy = checkpoint['entropy']
            performance_score = checkpoint['performance_score']

            print(f"Model loaded from {path}")
            print(f"Loaded model state: Episode {
                  episode}, Reward {reward:.2f}")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {
                  value_loss:.4f}, Entropy: {entropy:.4f}")
            print(f"Performance Score: {performance_score:.4f}")
            self.best_performance = performance_score
            self.best_reward = reward
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        running_return = 0
        running_advantage = 0

        for t in reversed(range(rewards.size(0))):
            running_return = rewards[t] + self.gamma * running_return * ~dones[t]
            returns[t] = running_return

            if t == rewards.shape[0] - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * ~dones[t] - values[t]
            running_advantage = delta + self.gamma * self.lambda_ * running_advantage * ~dones[t]
            advantages[t] = running_advantage

        return advantages, returns

    def compute_poca_advantages(self, advantages, values, actions, action_probs, observations, obs_masks, team_states, team_masks):
        batch_size, num_agents, _ = action_probs.shape
        poca_advantages = advantages.clone()

        for i in range(num_agents):
            # 現在のエージェント以外の行動をマスク
            masked_actions = actions.clone()
            masked_actions[:, i] = -1  # マスク値

            # 各行動に対する反実仮想値を計算
            cf_values = []
            for a in range(self.env.action_space):
                temp_actions = masked_actions.clone()
                temp_actions[:, i] = a
                _, cf_value = self.network.forward(
                    observations, obs_masks, team_states, team_masks)
                cf_values.append(cf_value[:, i])

            # [batch_size, action_space]の形状に変形
            cf_values = torch.stack(cf_values, dim=1)

            # 期待反実仮想値を計算
            expected_cf_value = (cf_values * action_probs[:, i]).sum(dim=1)[:-1]

            # POCAアドバンテージを計算
            poca_advantages[:, i] = advantages[:, i] - \
                (expected_cf_value - values[:-1, i])
        return poca_advantages

    def train(self, num_episodes: int, max_steps: int):
        for episode in range(num_episodes):
            observations, team_states = self.env.reset()
            episode_rewards = []
            episode_values = []
            episode_action_probs = []
            episode_actions = []
            episode_dones = []
            episode_observations = [observations[0]]
            episode_observation_masks = [observations[1]]
            episode_team_states = [team_states[0]]
            episode_team_state_masks = [team_states[1]]

            for step in range(max_steps):
                action_probs, values = self.network.forward(*observations, *team_states)
                action_probs = action_probs.squeeze(0)
                episode_action_probs.append(action_probs)
                episode_values.append(values.squeeze(0))
                actions = torch.multinomial(action_probs, 1)
                episode_actions.append(actions)

                next_observations, next_team_states, rewards, dones = self.env.step(
                    actions.numpy())

                episode_rewards.append(rewards)
                episode_dones.append(dones)

                episode_observations.append(next_observations[0])
                episode_observation_masks.append(next_observations[1])
                episode_team_states.append(team_states[0])
                episode_team_state_masks.append(team_states[1])

                if all(dones):
                    break

                observations, team_states = next_observations, next_team_states

            # エピソード終了後の処理
            _, final_values = self.network.forward(*observations, *team_states)
            episode_action_probs.append(action_probs.squeeze(0))
            episode_values.append(final_values.squeeze(0))
            episode_rewards = torch.stack(episode_rewards)
            episode_values = torch.stack(episode_values)
            episode_action_probs = torch.stack(episode_action_probs)
            episode_actions = torch.stack(episode_actions)
            episode_dones = torch.stack(episode_dones)
            episode_observations = torch.cat(episode_observations,dim=0)
            episode_observation_masks = torch.cat(
                episode_observation_masks, dim=0)
            episode_team_states = torch.cat(episode_team_states, dim=0)
            episode_team_state_masks = torch.cat(
                episode_team_state_masks, dim=0)

            advantages, returns = self.compute_advantages(
                episode_rewards, episode_values, episode_dones)
            poca_advantages = self.compute_poca_advantages(
                advantages, episode_values, episode_actions, episode_action_probs,
                episode_observations, episode_observation_masks, episode_team_states, episode_team_state_masks)

            policy_loss, value_loss, entropy = self.update_network(
                episode_observations, episode_observation_masks,
                episode_team_states, episode_team_state_masks,
                episode_action_probs.detach(), poca_advantages.unsqueeze(-1).detach(), returns.detach())

            episode_reward = episode_rewards.sum().item()

            print("Episode {}, Total Reward: {}".format(episode, episode_reward))
            print("\tPolicy Loss: {:.4f}\n\tValue Loss: {:.4f}\n\tEntropy: {:.4f}".format(
                policy_loss, value_loss, entropy
            ))

            if (episode + 1) % self.save_interval == 0:
                performance_score = self.save_model(episode + 1,
                                episode_reward, policy_loss, value_loss, entropy)
            else:
                performance_score = self.calc_performance_score(
                    episode_reward, policy_loss, value_loss, entropy)
            # 最高性能を更新した場合
            if performance_score > self.best_performance:
                self.best_performance = performance_score
                print(f"New best performance: {performance_score:.4f}")
                self.save_model(episode + 1,
                                episode_reward, policy_loss, value_loss, entropy)

    def update_network(self, observations, observation_masks, team_states, team_state_masks, old_action_probs, advantages, returns):
        # Flatten the batch and agent dimensions
        batch_size, num_agents = advantages.shape[:2]
        flat_old_action_probs = old_action_probs[:-1].view(batch_size * num_agents, -1)
        flat_advantages = advantages.view(-1,1)
        flat_returns = returns.view(-1, 1)

        for _ in range(self.num_epochs):
            # Generate new action probabilities and state values
            new_action_probs, new_values = self.network(
                observations, observation_masks,
                team_states, team_state_masks
            )
            new_action_probs = new_action_probs[:-1].view(batch_size * num_agents, -1)
            new_values = new_values[:-1].view(-1, 1)

            # Compute probability ratio
            ratio = new_action_probs / (flat_old_action_probs + 1e-8)

            # Compute surrogate losses
            surrogate1 = ratio * flat_advantages
            surrogate2 = torch.clamp(
                ratio, 1 - self.clip_param, 1 + self.clip_param) * flat_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Compute value loss using clipped value estimates
            value_pred_clipped = flat_old_action_probs + \
                (new_values - flat_old_action_probs).clamp(-self.clip_param, self.clip_param)
            value_losses = (new_values - flat_returns).pow(2)
            value_losses_clipped = (value_pred_clipped - flat_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            # Compute entropy bonus
            entropy = -(new_action_probs *
                        torch.log(new_action_probs + 1e-8)).sum(dim=-1).mean()

            # Compute total loss
            loss = policy_loss + self.value_loss_coef * \
                value_loss - self.entropy_coef * entropy

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()


class EnvVisualizer:
    def __init__(self, env: GameEnvironment, window_size=(800, 600), fps=30):
        self.env = env
        self.window_size = window_size
        self.scale = min(window_size[0] / env.map_size[0],
                         window_size[1] / env.map_size[1])
        self.fps = fps
        self.player_size = 10

        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Game Environment Visualization")

        self.colors = [
            (255, 0, 0),   # Red
            (0, 0, 255),   # Blue
            (0, 255, 0),   # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        self.font = pygame.font.Font(None, 24)  # フォントの初期化

        self.fov_surface = pygame.Surface(window_size, pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

    def draw(self):
        self.screen.fill((255, 255, 255))  # White background
        self.fov_surface.fill((0, 0, 0, 0))  # Clear the FOV surface

        for i, player in enumerate(self.env.players):
            color = self.colors[player['team'] % len(self.colors)]
            pos = self._to_screen_coords(player['position'])
            direction = player['direction'] * self.env.TO_RAD

            if player['hp'] > 0:
                # 生存しているプレイヤーの描画
                self.draw_fov_sector(pos, direction, (*color, 20))

                # プレイヤーの三角形を描画
                points = [
                    (pos[0] + self.player_size * math.cos(direction),
                     pos[1] + self.player_size * math.sin(direction)),
                    (pos[0] + self.player_size/2 * math.cos(direction + 2.6),
                     pos[1] + self.player_size/2 * math.sin(direction - 2.6)),
                    (pos[0] + self.player_size/2 * math.cos(direction - 2.6),
                     pos[1] + self.player_size/2 * math.sin(direction + 2.6))
                ]
                pygame.draw.polygon(self.screen, color, points)

                # 体力バーを描画
                health_width = 20
                health_height = 5
                health_pos = (pos[0] - health_width // 2, pos[1] - 20)
                pygame.draw.rect(self.screen, (255, 0, 0),
                                 (*health_pos, health_width, health_height))
                pygame.draw.rect(self.screen, (0, 255, 0), (*health_pos,
                                 health_width * player['hp'] / self.env.MAX_HP, health_height))

                # 残弾数を描画
                ammo_text = self.font.render(
                    f"{player['ammo']:.1f}", True, (0, 0, 0))
                ammo_pos = (pos[0] - ammo_text.get_width() // 2, pos[1] + 15)
                self.screen.blit(ammo_text, ammo_pos)
            else:
                # 死亡したプレイヤーの位置にバツ印を描画
                self.draw_x_mark(pos, color)

        # Draw bullets
        for bullet in self.env.bullets:
            pos = self._to_screen_coords(bullet['position'])
            pygame.draw.circle(self.screen, (0, 0, 0), pos, 2)

        # FOVサーフェスをメインスクリーンにブリット
        self.screen.blit(self.fov_surface, (0, 0))

        pygame.display.flip()
        self.clock.tick(self.fps)  # FPSに基づいてフレームレートを制御

    def draw_fov_sector(self, pos, direction, color):
        fov_angle = np.pi  # 90度のFOV
        fov_radius = 100  # FOVの半径
        num_points = 50  # 扇形の滑らかさを決定する点の数

        points = [pos]
        for i in range(num_points + 1):
            angle = direction - fov_angle / 2 + fov_angle * i / num_points
            x = pos[0] + fov_radius * math.cos(angle)
            y = pos[1] + fov_radius * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(self.fov_surface, color, points)

    def draw_x_mark(self, pos, color):
        size = self.player_size
        pygame.draw.line(self.screen, color, (pos[0] - size, pos[1] - size),
                         (pos[0] + size, pos[1] + size), 2)
        pygame.draw.line(self.screen, color, (pos[0] + size, pos[1] - size),
                         (pos[0] - size, pos[1] + size), 2)

    def _to_screen_coords(self, pos):
        return (int(pos[0] * self.scale), int(pos[1] * self.scale))

    def close(self):
        pygame.quit()


def display_game_scores(env: GameEnvironment):
    team_stats = {}
    player_stats = env.get_player_stats()

    # チーム統計の初期化
    for team in range(env.num_teams):
        team_stats[team] = {
            'kills': 0,
            'damage_dealt': 0,
            'team_attacks': 0,
            'team_kills': 0,
            'shots_fired': 0,
            'shots_hit': 0,
            'survivors': 0,
            'total_hp': 0,
            'score': 0
        }

    # プレイヤー統計からチーム統計を計算
    for player, stats in zip(env.players, player_stats):
        team = stats['team']
        team_stats[team]['kills'] += stats['kills']
        team_stats[team]['damage_dealt'] += stats['damage_dealt']
        team_stats[team]['team_attacks'] += stats['team_attack']
        team_stats[team]['team_kills'] += stats['team_kill']
        team_stats[team]['shots_fired'] += stats['shots_fired']
        team_stats[team]['shots_hit'] += stats['shots_hitted']
        team_stats[team]['survivors'] += int(stats['survived'])
        team_stats[team]['total_hp'] += player['hp']

    # スコア計算
    for team, stats in team_stats.items():
        stats['score'] = (
            stats['kills'] * 100 +
            stats['damage_dealt'] +
            stats['survivors'] * 50 +
            stats['total_hp'] -
            stats['team_attacks'] * 10 -
            stats['team_kills'] * 100
        )

    # チーム別スコアテーブルの作成
    team_headers = ["Team", "Kills", "Damage", "Team Attacks",
                    "Team Kills", "Accuracy", "Survivors", "Total HP", "Score"]
    team_table_data = []

    for team, stats in team_stats.items():
        accuracy = (stats['shots_hit'] / stats['shots_fired']
                    * 100) if stats['shots_fired'] > 0 else 0
        row = [
            colored(f"Team {team}", env.team_colors[team]),
            stats['kills'],
            stats['damage_dealt'],
            stats['team_attacks'],
            stats['team_kills'],
            f"{accuracy:.1f}%",
            stats['survivors'],
            stats['total_hp'],
            stats['score']
        ]
        team_table_data.append(row)

    # テーブルの表示
    print("\n" + "=" * 80)
    print(colored("Team Scores", "yellow", attrs=['bold']))
    print(tabulate(team_table_data, headers=team_headers, tablefmt="fancy_grid"))
    print("=" * 80 + "\n")

    # プレイヤー別スコアテーブルの作成
    player_headers = ["Player", "Team", "Kills", "Damage",
                      "Team Attacks", "Team Kills", "Accuracy", "Survived"]
    player_table_data = []

    for i, (player, stats) in enumerate(zip(env.players, player_stats)):
        accuracy = (stats['shots_hitted'] / stats['shots_fired']
                    * 100) if stats['shots_fired'] > 0 else 0
        row = [
            f"Player {i}",
            colored(f"Team {stats['team']}", env.team_colors[stats['team']]),
            stats['kills'],
            stats['damage_dealt'],
            stats['team_attack'],
            stats['team_kill'],
            f"{accuracy:.1f}%",
            "Yes" if stats['survived'] else "No"
        ]
        player_table_data.append(row)

    print(colored("Player Statistics", "yellow", attrs=['bold']))
    print(tabulate(player_table_data, headers=player_headers, tablefmt="fancy_grid"))
    print("=" * 80 + "\n")

    # 勝利チームの表示
    max_score = max(stats['score'] for stats in team_stats.values())
    winning_teams = [team for team, stats in team_stats.items()
                     if stats['score'] == max_score]

    if len(winning_teams) == 1:
        winner = winning_teams[0]
        print(colored(f"🏆 Team {winner} wins with a score of {
              max_score}! 🏆", env.team_colors[winner], attrs=['bold']))
    else:
        print(colored("🏆 It's a tie between: " + ", ".join(f"Team {
              team}" for team in winning_teams) + f" with a score of {max_score}! 🏆", "yellow", attrs=['bold']))

def visualize_game(env, model, num_steps=1000):
    visualizer = EnvVisualizer(env)
    observations, team_states = env.reset()

    for _ in range(num_steps):
        visualizer.draw()

        action_probs, _ = model(
            observations[0], observations[1], team_states[0], team_states[1])
        actions = torch.multinomial(action_probs.squeeze(0), 1)

        observations, team_states, rewards, dones = env.step(actions.numpy())

        if all(dones):
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                visualizer.close()
                return

    visualizer.close()
    display_game_scores(visualizer.env)  # ゲーム終了後にスコアを表示


# 使用方法
load_best_model = True
train = False
visualize = True
env = GameEnvironment(num_teams=2, players_per_team=3, map_size=(50, 50))
model = AgentNetwork(*env.get_dims())
trainer = MAPOCA(env, model)
if load_best_model:
    if trainer.load_best_model():
        print("Best model loaded successfully. Ready for evaluation or further training.")
    else:
        print("Failed to load best model. Starting with a fresh model.")
if train:
    trainer.train(50, 1000)
if visualize:
    visualize_game(env, trainer.network)
