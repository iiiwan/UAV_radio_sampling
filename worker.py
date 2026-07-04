import os
import copy
import torch
import imageio
import numpy as np

from env import Env
from parameters import *
import scipy.signal as signal
import matplotlib.pyplot as plt
from attention_net import AttentionNet

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):

        # 设备
        self.device = device
        # 是否贪心探索
        self.greedy = greedy
        # 当前启动的环境编号
        self.metaAgentID = metaAgentID
        #
        self.global_step = global_step
        # 是否保存图片
        self.save_image = save_image
        #
        self.sample_length = sample_length
        # 采样点数量
        self.sample_size = sample_size

        # 载入环境
        self.env = Env(sample_size=self.sample_size, k_size=K_SIZE, budget_range=budget_range,
                       save_image=self.save_image)
        # 载入网络
        self.local_net = localNetwork
        # 历史经验值
        self.experience = None

    def run_episode(self, currEpisode):
        # 回放经验存储
        episode_buffer = []
        perf_metrics = dict()
        for i in range(13):
            episode_buffer.append([])

        # 是否能耗耗完标志
        done = False
        # 初始化环境， 获得当前动态图的形式和每个点的方差、值
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        # 初始化无人车路径
        route = self.env.route_coords.copy()
        # 输入网络特征
        n_nodes = node_coords.shape[0]
        # 点对应的预测值
        node_info_inputs = node_info.reshape((n_nodes, 1))
        # 点对应的方差
        node_std_inputs = node_std.reshape((n_nodes, 1))
        # 当前位置到图中每个位置的能耗
        budget_inputs = self.calc_estimate_budget(budget, current_idx=0 )
        node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 1)

        # 把每个点能连接的点输出保存
        graph = list(graph.values())
        edge_inputs = []
        # 遍历graph中的每一个节点信息
        for node in graph:
            # 将当前节点node的相邻节点转换为整型，并将它们放入一个列表中
            node_edges = list(map(int, node))
            # print(f'节点的相邻节点个数为: {len(node_edges)}')
            # 将当前节点的所有相邻节点列表添加到edge_inputs中
            edge_inputs.append(node_edges)

        # 计算边相连信息
        edge_index = []
        for node_id, neighbors in enumerate(graph):
            for neighbor in neighbors:
                edge_index.append([node_id, int(neighbor)])
        # 转换成 PyTorch 张量，并进行转置，以符合 GCN 的输入格式
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)

        # 计算位置嵌入向量
        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)  # (1, sample_size+2, 32)
        # 图的边嵌入向量
        # unsqueeze(0) 的作用是为 edge_inputs 增加一个新的维度，这个维度通常用于表示批次
        # 某些节点的连接数量不足 k_size，那么会用默认值或 padding 填充
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, k_size)
        # 当前位置
        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        # 初始化模型参数
        LSTM_h = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        # 初始化mask
        mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

        # 开始与环境交互
        for i in range(256):
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding

            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs,
                                                                  current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            # next_node (1), logp_list (1, 10), value (1,1,1)
            # 如果是贪心选择，则一直选择最大的
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            episode_buffer[8] += budget_inputs

            # 获取网络输出的下一个点序号
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            # 将该动作与环境交互
            reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            # 更新当前的图结构和点坐标
            graph, node_coords = self.env.graph, self.env.node_coords
            # 更新无人车路径
            route = self.env.route_coords.copy()

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)

            # 把每个点能连接的点输出保存
            graph = list(graph.values())
            edge_inputs = []
            # 遍历graph中的每一个节点信息
            for node in graph:
                # 将当前节点node的相邻节点转换为整型，并将它们放入一个列表中
                node_edges = list(map(int, node))
                # print(f'节点的相邻节点个数为: {len(node_edges)}')
                # 将当前节点的所有相邻节点列表添加到edge_inputs中
                edge_inputs.append(node_edges)
            edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, k_size)

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

            # 保存每一次交互的路径图
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot(route, self.global_step, i, gifs_path)

            # 如果能耗耗完
            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                perf_metrics['remain_budget'] = remain_budget / budget
                perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                perf_metrics['cov_trace'] = self.env.cov_trace
                perf_metrics['success_rate'] = True
                print('{} Goodbye world! We did it!, remain_budget is {}'.format(i, remain_budget))
                print('route is ', self.env.route)
                break
        # 如果交互256次都没有耗完能耗
        if not done:
            episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index,
                                                          LSTM_h, LSTM_c, pos_encoding, mask)
            episode_buffer[6].append(value.squeeze(0))
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
            perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['success_rate'] = False
            perf_metrics['sum_reward'] = self.env.sum

        REWARD.append(self.env.sum)
        plt.figure()
        # 绘制奖励曲线
        plt.plot(REWARD, label='Reward')
        # 添加图形标题和标签
        plt.title('Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # 保存图像到本地
        plt.savefig('d=512.png')

        # 计算折扣奖励
        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward, dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)

        for i in range(target_v.size()[0]):
            episode_buffer[7].append(target_v[i, :, :])

        # 将每一次交互图保存为gif图
        if self.save_image:
            path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics

    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.node_coords[current_idx]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current2point, _ = self.env.prm.mycaldistance(current_coord, point_coord)
            estimate_budget = (budget - dist_current2point) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)

    def calculate_position_embedding(self, edge_inputs):
        # 邻接矩阵
        A_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        # 度矩阵
        D_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        # 构建邻接矩阵
        for i in range(self.sample_size+2):
            for j in range(self.sample_size+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        # 构建度矩阵
        for i in range(self.sample_size+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        # 计算归一化图拉普拉斯矩阵
        L = np.eye(self.sample_size+2) - np.matmul(D_matrix, A_matrix, D_matrix)
        # 使用 np.linalg.eig(L) 对拉普拉斯矩阵 L 进行特征分解，得到特征值 eigen_values 和特征向量 eigen_vector
        eigen_values, eigen_vector = np.linalg.eig(L)
        # argsort() 用于对特征值进行升序排序，并通过 idx 对特征向量也进行相应的排序。
        # 只保留特征向量的实部 np.real()，因为某些特征分解可能会产生复数值
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        # 去掉第一个特征向量（对应特征值为零，通常为常量向量），然后选择第2到第33个特征向量（总共32个）。
        # 这些特征向量通常捕捉了图结构的局部信息，可以用于生成节点的位置嵌入。
        eigen_vector = eigen_vector[:,1:32+1]
        if eigen_vector.shape[1] < 32:
            # 使用零填充到 32 维
            padding = np.zeros((eigen_vector.shape[0], 32 - eigen_vector.shape[1]))
            eigen_vector = np.hstack((eigen_vector, padding))
        return eigen_vector

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')
        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)