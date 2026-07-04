import numpy as np
import copy

from parameters import *
from itertools import product
from matplotlib import pyplot as plt
from gp_ipp import GaussianProcessForIPP
from classes.Gaussian2D import Gaussian2D
from classes.graph_generater import graph_generater


class Env():
    def __init__(self, sample_size=40, k_size=10, start=None, destination=None, obstacle=[], budget_range=None, save_image=False, seed=None):
        # 地图大小
        self.length = 100
        # 周围采样点个数
        self.sample_size = sample_size
        #
        self.k_size = k_size
        # 能耗生成范围
        self.budget_range = budget_range
        # 生成无人车能耗
        self.budget = np.random.uniform(*self.budget_range)
        # 障碍物信息
        self.obstacle = obstacle
        # 生成无人车终点
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])
        # 无人车轨迹坐标
        self.route_coords = []
        # 设定随机种子
        self.seed = seed
        # 确定当前无人车位置的序号
        self.current_node_index = 0

        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None
        self.high_info_area = None

        # GP
        self.gp_ipp = None
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy((self.node_info, self.node_std, self.budget))
        self.RMSE = None
        self.F1score = None
        self.cov_trace = None
        self.MI = None
        self.MI0 = None

        # start point
        self.current_node_index = 0
        self.dist_residual = 0
        self.route = []
        # 计算总奖励
        self.sum = 0

        self.save_image = save_image
        self.frame_files = []


    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # 生成真实电磁地图
        self.underlying_distribution = Gaussian2D()
        self.ground_truth, self.obstacle = self.get_ground_truth()

        # 生成无人车起点
        while True:
            # 随机生成起点，假设坐标在 [0, 1) 范围内
            start = np.random.rand(1, 2)
            # 将 [0, 1) 范围内的坐标映射到障碍物地图的 50*50 网格
            start_grid_x = int(start[0][0] * self.length)
            start_grid_y = int(start[0][1] * self.length)
            # 检查是否是障碍物
            if self.obstacle[start_grid_x][start_grid_y] == 0:  # 0 表示不是障碍物
                self.start = start
                break
            else:
                # 重新生成 start，直到找到不在障碍物的区域
                continue

        # 初始化动态图的生成器
        self.prm = graph_generater(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range,
                                   self.k_size)
        self.node_coords, self.graph = self.prm.create_graph(saveImage=False, seed=seed, currocoord=self.start,
                                                             curridx=self.current_node_index)

        # 初始化高斯过程
        self.gp_ipp = GaussianProcessForIPP(self.node_coords)
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        # 预测当前动态图上的点的方差和值
        self.node_info, self.node_std = self.gp_ipp.update_node()

        # for i in range(len(self.node_info)):
        #     value, _ = self.underlying_distribution.distribution_function(
        #         self.node_coords[i].reshape(-1, 2))
        #     self.node_info[i] = value


        # 初始化各项指标
        self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        self.MI = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        self.cov_trace0 = self.cov_trace

        # 保存最初始状态
        self.node_info0, self.node_std0, self.budget = copy.deepcopy((self.node_info, self.node_std, self.budget0))

        # start point
        self.current_node_index = 0
        self.sample = self.start
        self.dist_residual = 0
        self.route = []
        self.route.append(self.current_node_index)
        self.route_coords = [self.node_coords[self.current_node_index]]
        self.sum = 0
        np.random.seed(None)

        return self.node_coords, self.graph, self.node_info, self.node_std, self.budget

    def step(self, next_node_index, sample_length, measurement=True):
        # 计算下一个目标点和当前位置之间的距离
        # if str(self.current_node_index) in self.graph and str(next_node_index) in self.graph[str(self.current_node_index)]:
        #     if self.graph[str(self.current_node_index)][str(next_node_index)].length == 999:
        #         cost = 999
        #     else:
        #         cost = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[next_node_index])
        # else:
        #     cost = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[next_node_index])
        #
        # if cost == 999:
        #     reward = -0.01
        #     done = False
        #     return reward, done, self.node_info, self.node_std, self.budget

        # 03
        # self.prm.checkLine(self.node_coords[self.current_node_index] , self.node_coords[next_node_index], self.current_node_index, next_node_index)
        dist, path_to_next = self.prm.mycaldistance(self.node_coords[self.current_node_index] , self.node_coords[next_node_index])
        path_to_next = [int(x) for x in path_to_next]
        # 将 path_to_next 的第一个点跳过，剩余点添加到 self.route_coords
        self.route_coords.extend([self.node_coords[point] for point in path_to_next[1:]])

        remain_length = dist
        next_length = sample_length - self.dist_residual
        reward = 0

        done = True if self.budget <= 0 else False

        no_sample = True
        # while remain_length > next_length:
        #     if no_sample:
        #         self.sample = (self.node_coords[next_node_index] - self.node_coords[
        #             self.current_node_index]) * next_length / dist + self.node_coords[self.current_node_index]
        #     else:
        #         self.sample = (self.node_coords[next_node_index] - self.node_coords[
        #             self.current_node_index]) * next_length / dist + self.sample
        #     if measurement:
        #         observed_value,_ = self.underlying_distribution.distribution_function(
        #             self.sample.reshape(-1, 2))
        #         observed_value = observed_value + np.random.normal(0, 1e-10)
        #     else:
        #         observed_value = np.array([0])
        #     self.gp_ipp.add_observed_point(self.sample, observed_value)
        #
        #     remain_length -= next_length
        #     next_length = sample_length
        #     no_sample = False
        observed_value,_ = self.underlying_distribution.distribution_function(
            self.node_coords[next_node_index].reshape(-1, 2))
        observed_value = observed_value + np.random.normal(0, 1e-10)

        d = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[next_node_index])
        r2 = abs(self.node_info[self.current_node_index] - self.node_info[next_node_index])/d
        self.gp_ipp.add_observed_point(self.node_coords[next_node_index], observed_value)

        # 更新高斯过程
        self.gp_ipp.update_gp()

        # 生成新的动态图
        self.node_coords, self.graph = self.prm.create_graph(saveImage=False, currocoord=self.node_coords[next_node_index], curridx= next_node_index)

        # 将高斯过程中的预测点进行更新，如果是k近邻更新，那么需在计算迹之前重新update_gp一下
        self.gp_ipp.new_coords(self.node_coords)
        self.node_info, self.node_std = self.gp_ipp.update_node()
        # for i in range(len(self.node_info)):
        #     value, _ = self.underlying_distribution.distribution_function(
        #         self.node_coords[i].reshape(-1, 2))
        #     self.node_info[i] = value
        # k-GP所需要的
        self.gp_ipp.update_gp()

        if measurement:
            self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
            # F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
            RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
            self.RMSE = RMSE
        cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)

        # 重复采样则奖励为负
        if next_node_index in self.route[-2:]:
            reward += -0.1

        # 不确定性减少
        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
            reward += r2 * 0.02

            # print('reward is', reward)
            # print('r2 is', r2 * 0.05)

        self.cov_trace = cov_trace

        if done:
            reward -= cov_trace / (self.length * self.length)
            # reward -= cov_trace / 10

        self.sum = self.sum + reward

        # self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        self.budget -= dist
        self.current_node_index = next_node_index
        self.route.append(next_node_index)

        return reward, done, self.node_info, self.node_std, self.budget


    def get_ground_truth(self):
        x1 = np.linspace(0, 1, self.length)
        x2 = np.linspace(0, 1, self.length)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth, obs_map = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth, obs_map

    def plot(self, route, n, step, path, testID=0, CMAES_route=True, sampling_path=False):
        # Plotting shorest path
        plt.switch_backend('agg')
        self.gp_ipp.plot(self.ground_truth, self.obstacle)
        if CMAES_route:
            pointsToDisplay = route
        else:
            pointsToDisplay = [(self.prm.findPointsFromNode(path)) for path in route]
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=1, zorder=5)
        if sampling_path:
            pointsToDisplay2 = [(self.prm.findPointsFromNode(path)) for path in sampling_path]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(x0[i:i + 2], y0[i:i + 2], c='white', linewidth=4, zorder=5, alpha=1- 0.2 * i / len(x0))
            for i in range(len(x1) - 1):
                plt.plot(x1[i:i + 2], y1[i:i + 2], c='red', linewidth=4, zorder=6)

        # for i in range(len(x)-1):
        #     plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        plt.suptitle('Budget: {:.4g}/{:.4g},  Cov trace: {:.4g}'.format(
            self.budget, self.budget0, self.cov_trace))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size), dpi=150)
        # plt.show()
        frame = '{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size)
        self.frame_files.append(frame)