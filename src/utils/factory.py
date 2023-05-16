from typing import List, Callable
import numpy as np
import torch
from torch import tensor
from botorch.test_functions import Rastrigin, Rosenbrock, Ackley, Levy, DixonPrice
from matplotlib import pyplot as plt

import os


from .general import sample_pts, feasible_filter_gen
from botorch.utils.transforms import unnormalize

device = torch.device('cpu')
dtype = torch.float

class Data_Factory:
    """
    Collections of different objective functions
    """

    def __generate_config(self, dim, num) -> None:
        """Generate required num & dim config following uniform distribution"""
        if dim == 1:
            self.config = np.random.uniform(low=[-1], high=[1], size=[num, dim])
            self.config = np.hstack([self.config, np.zeros([num, dim])])
        elif dim > 1:
            self.config = np.random.uniform(
                low=-1 * np.ones(dim), high=1 * np.ones(dim), size=[num, dim]
            )

    def convex_1(self, dim: int = 3, num: int = 1000) -> np.ndarray:
        """
        simple d-dim convex function
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = np.sum(self.config ** 2, axis=1)[:, np.newaxis]
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def convex_2(self, dim: int = 3, num: int = 1000) -> np.ndarray:
        """
        simple d-dim convex function for optimization
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = -np.sum(self.config ** 2, axis=1)[:, np.newaxis]
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def example_fn(self, dim: int = 3, num: int = 1000) -> np.ndarray:
        """
        f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = np.sum(
            np.sin(10 * self.config) * (np.exp(-self.config) ** 2), axis=1
        )
        self.data = np.hstack([self.config, self.target_value[:, np.newaxis]])
        return self.data

    def exmaple_fn_1d(self, dim: int = 1, num: int = 1000) -> np.ndarray:
        """DKL 1d example"""
        assert dim == 1
        self.__generate_config(dim=dim, num=num)
        x = self.config
        self.target_value = (x + 0.5 >= 0) * np.sin(64 * (x + 0.5) ** 4)
        self.data = np.hstack(
            [self.config, self.target_value[:, 0].reshape([num, dim])]
        )
        return self.data

    def robot_arm_8d(self, num: int = 10000) -> np.ndarray:
        """https://www.sfu.ca/~ssurjano/robot.html"""
        assert num >= 1
        dim = 8
        self.__generate_config(dim=dim, num=num)
        self.config = (self.config + 1) / 2
        self.config[:, :4] = self.config[:, :4] * np.pi
        u = np.zeros([num, 1])
        v = np.zeros([num, 1])
        for i in range(4):
            tmp = np.sum(self.config[:, : i + 1], axis=1).reshape([num, 1])
            u += self.config[:, i + 4].reshape([num, 1]) * np.cos(tmp)
            v += self.config[:, i + 4].reshape([num, 1]) * np.sin(tmp)
        self.target_value = np.sqrt(u ** 2 + v ** 2)
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def rastrigin(self, dim=4, num: int = 1000) -> np.ndarray:
        """https://www.sfu.ca/~ssurjano/rastr.html"""
        assert dim > 1
        self.__generate_config(dim=dim, num=num)
        self.config = self.config * 5.12
        self.tmp = self.config ** 2 - 10 * np.cos(2 * np.pi * self.config)
        self.target_value = 10 * dim + np.sum(self.tmp, axis=1).reshape([num, 1])
        # raise(Exception("Not implemented!"))
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    @staticmethod
    def nearest(np_data:np.ndarray, point: List) -> np:
        """
        Tool to find nearest data point
        """
        length = np.size(point)
        diff = np.abs(np_data[:, :length] - np.array(point))
        index = np.argmin(np.sum(diff, axis=1))
        return np_data[index], index

    def obj_func(self, test_p: list) -> float:
        """For query druing optimization"""
        input_dim = np.shape(test_p)[0]
        if input_dim != np.size(test_p) and input_dim > 1:
            value = [
                Data_Factory.nearest(self.data, test_p[i, :])[-1]
                for i in range(input_dim)
            ]
            return value
        else:
            data_point = Data_Factory.nearest(self.data, test_p)
            return data_point[-1]

    def plot_1Dsample(self):
        plt.scatter(self.data[:, 0], self.data[:, -1])
        plt.title("1-D Demo of Original Data")
        plt.xlabel("Config")
        plt.ylabel("Target Value")


class Constrained_Data_Factory(Data_Factory):
    """
    Collections of different objective functions, together with their constraints.
    For each test function, use sobel engine to sample x_tensor, 
     Return:
        x_tensor, or standardized_x_tensor
        y_tensor or f_func, 
        list of c_tensor or list of c_func for SCBO < 0
    """
    def __init__(self, num_pts:int = 20000) -> None:
        super().__init__()
        self.dtype = torch.float
        self.device = torch.device('cpu')
        self._num_pts = num_pts
    
    def _generate_x_tensor(self, dim:int, num:int, seed:int=0) -> tensor:
        '''
        return samples in [0, 1]^dim
        '''
        x_tensor = sample_pts(lb=torch.zeros(dim), ub=torch.ones(dim), n_pts=num,  dim=dim, seed=seed)
        return x_tensor.to(device=self.device, dtype=self.dtype)
    
    @staticmethod
    def evaluate_func(lb:tensor, ub:tensor, func:Callable, x_tensor:tensor) -> tensor:
        return torch.tensor([func(unnormalize(x, (lb, ub))) for x in x_tensor])
    
    @staticmethod
    def nearest_approx(data:tensor, point:tensor, reward_start_idx:int=32, reward_idx:int=-1) -> tensor:
        """
        Tool to find nearest data point (L1) and return corresponding value
        """
        diff = torch.abs(data[:, :reward_start_idx] - point)
        index = torch.argmin(diff.sum(dim=-1))
        return data[index, reward_idx]

    def rastrigin_1D(self, scbo_format=False) -> List[tensor]:
        """https://www.sfu.ca/~ssurjano/rastr.html"""
        self._name = 'Rastrigin 1D'
        dim = 1
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -5, torch.ones(dim) * 5
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        
        self.objective = lambda x: -Rastrigin(dim=1)(x)
        # self.c_func1 = lambda x: -(x+3)**2 + 0.64  # |x - -2| < 0.5
        self.c_func1 = lambda x: -torch.abs(x+3.1)**(1/2) + .8 **(1/2) 
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func_list = [self.c_func1_scbo]
        self.x_tensor = self._generate_x_tensor(dim=1, num=self._num_pts).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1]
        self.constraint_threshold_list = [0]
        self.constraint_confidence_list = [0.5]
        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)

        assert torch.any(self.feasible_filter)

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list
        
    def ackley_5d(self, scbo_format=False) -> List[tensor]:
        '''
        Note: due to earlier problem, it is actually 4d. but feeded to a 10 d function. and only on diagonal.
        '''
        self._name = 'Ackely'
        dim = 4
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -5, torch.ones(dim) * 3
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        self.objective = lambda x: Ackley(dim=10)(x) # deliberately set to be 10, still work on dim = 1
        self.c_func1 = lambda x: -torch.sum(x) # sum x <= 0
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func2 = lambda x: - torch.linalg.vector_norm(x-torch.ones(dim)) + 4 # norm x < 5
        self.c_func2_scbo = lambda x: -self.c_func2(x)
        self.c_func_list = [self.c_func1_scbo, self.c_func2_scbo]
        self.x_tensor = self._generate_x_tensor(dim=1, num=self._num_pts, seed=2).to(device=device, dtype=dtype)
        self.x_tensor = unnormalize(self.x_tensor, (torch.zeros(dim).to(device=device, dtype=dtype), torch.ones(dim).to(device=device, dtype=dtype)))
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor2 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func2, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1, self.c_tensor2]
        self.constraint_threshold_list = [0, 0]
        self.constraint_confidence_list = [0.5, 0.5]
        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)

        assert torch.any(self.feasible_filter)

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list

    def rosenbrock_5d(self, scbo_format=False) -> List[tensor]:
        self._name = 'Rosenbrock_3C'
        dim = 2
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -3, torch.ones(dim) * 5
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        self.objective = lambda x: -0.5 * Ackley(dim=dim)(x) # deliberately set to be 10, still work on dim = 1
        self.c_func1 = lambda x: -DixonPrice(dim=dim)(x)/(dim*100) + .1
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func2 = lambda x: -Levy(dim=dim)(x)/(dim*100) + .1
        self.c_func2_scbo = lambda x: -self.c_func2(x)
        self.c_func3 = lambda x: torch.linalg.vector_norm(x, dim=-1)**(1/2) - (1.3)**(1/2)
        self.c_func3_scbo = lambda x: -self.c_func3(x)
        self.c_func_list = [self.c_func1_scbo, self.c_func2_scbo, self.c_func3_scbo]
        self.x_tensor = self._generate_x_tensor(dim=dim, num=self._num_pts, seed=0).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor2 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func2, self.x_tensor).unsqueeze(-1)
        self.c_tensor3 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func3, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1, self.c_tensor2, self.c_tensor3]
        self.constraint_threshold_list = [0, 0, 0]
        self.constraint_confidence_list = [0.5, 0.5, 0.5]
        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)

        assert torch.any(self.feasible_filter)
        print(f"Name {self._name} feasible pts {self.feasible_filter.sum()} over {self.feasible_filter.size(0)}")

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list

    def rosenbrock_4d(self, scbo_format=False) -> List[tensor]:
        self._name = 'Rosenbrock_2C'
        dim = 10
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -3, torch.ones(dim) * 5
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        self.objective = lambda x: -0.5 * Ackley(dim=dim)(x) # deliberately set to be 10, still work on dim = 1
        # self.c_func1 = lambda x: -torch.linalg.vector_norm(x, dim=-1)**(1/2) + (5)**(1/2)
        self.c_func1 = lambda x: -torch.linalg.vector_norm(x, dim=-1)**(1/2) + (8)**(1/2)
        # self.c_func1 = lambda x: torch.prod(x, dim=-1)**(1/dim)
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        # self.c_func2 = lambda x: torch.linalg.vector_norm(x, dim=-1)**(1/2) - (1.3)**(1/2)
        self.c_func2 = lambda x: torch.prod(torch.abs(x[:2]), dim=-1)**(1/3) - 1 ** (1/2)
        self.c_func2_scbo = lambda x: -self.c_func2(x)
        self.c_func_list = [self.c_func1_scbo, self.c_func2_scbo]
        self.x_tensor = self._generate_x_tensor(dim=dim, num=self._num_pts, seed=0).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor2 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func2, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1, self.c_tensor2]
        self.constraint_threshold_list = [0, 0]
        self.constraint_confidence_list = [0.5, 0.5]
        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)

        assert torch.any(self.feasible_filter)
        print(f"Name {self._name} feasible pts {self.feasible_filter.sum()} over {self.feasible_filter.size(0)}")

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list

    def water_converter_32d(self, scbo_format=False) -> List[tensor]:
        '''
        All subreward > 87682.7047
        '''
        self._name = "Water_Converter_16C"
        self.dim = 32
        _cali_factor = 10
        raw_threshold = 87000
        c_num = 2
        raw_factor = 109000
        _data_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../data/Sydney_Data.csv"
        raw_data = np.loadtxt(_data_path, delimiter=',')[:self._num_pts]
        data = torch.from_numpy(raw_data).to(device=device, dtype=dtype)
        self.x_tensor_range = data[:,:32]
        self.lb, self.ub = self.x_tensor_range.min(dim=0).values, self.x_tensor_range.max(dim=0).values
        self.x_tensor = (self.x_tensor_range - self.lb) / (self.ub - self.lb)
        self.y_tensor = data[:,-1].reshape([-1, 1]) / (raw_factor/_cali_factor) - 13 * _cali_factor
        raw_rewards = torch.from_numpy(raw_data[:,-17:-1]).to(device=device, dtype=dtype)
        self.objective = lambda x: Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=-1) / (raw_factor/_cali_factor) - _cali_factor
        self.c_tensor_list = [(raw_rewards[:, c_idx].reshape([-1,1]) - raw_threshold)/raw_factor for c_idx in range(c_num)]
        self.c_func_list = [lambda x: -(Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=c_idx+32)- raw_threshold)/raw_factor for c_idx in range(c_num)]
        self.constraint_threshold_list = [0 for _ in range(c_num)]
        self.constraint_confidence_list = [0.5 for _ in range(c_num)]

        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)
        # self.feasible_filter = self.feasible_filter.unsqueeze(0)
        assert torch.any(self.feasible_filter)
        print(f"Name {self._name} feasible pts {self.feasible_filter.sum()} over {self.feasible_filter.size(0)}")

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list

    def visualize_1d(self, if_norm:bool=False):
        fontsize = 25
        plt.figure(figsize=[12, 10])
        plt.title(self._name, fontsize=fontsize)
        if if_norm:
            base_x = torch.linalg.vector_norm(self.x_tensor_range, dim=-1)
        else:
            base_x = self.x_tensor_range
        plt.scatter(base_x.squeeze().to(device='cpu').numpy(), self.y_tensor.squeeze().to(device='cpu').numpy(), c='black', s=1, label='Objective')
        feasible_x = base_x[self.feasible_filter].to(device='cpu')
        feasible_y = self.y_tensor[self.feasible_filter].to(device='cpu')
        bounds = [feasible_x.min().to(device='cpu'), feasible_x.max().to(device='cpu')]
        plt.scatter(feasible_x.squeeze().to(device='cpu').numpy(), feasible_y.squeeze().to(device='cpu').numpy(), c='purple', s=1, label='Feasible region')
        # plt.vlines(x = bounds, ymin=self.y_tensor.min(), ymax=self.maximum, color='blue', label='Feasible region')
        plt.scatter(base_x[self.max_arg].to(device='cpu').numpy(), self.y_tensor[self.max_arg].to(device='cpu').numpy(), c='red', s=100, marker='*', label='Optimum' )
        plt.legend(fontsize=fontsize/1.4)
        plt.xlabel('X')
        plt.ylabel("Y")
        plt.savefig(f"./res/illustration/{self._name}")
        plt.close()


