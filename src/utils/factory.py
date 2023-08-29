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
        """For query during optimization"""
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
    def __init__(self, num_pts:int = 20000, venue:str='tmlr') -> None:
        super().__init__()
        self.dtype = torch.float
        self.device = torch.device('cpu')
        self._num_pts = num_pts
        self._venue = 'tmlr'
    
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

    def rastrigin_1D_1C(self, scbo_format:bool=False, **kwargs) -> List[tensor]:
        """https://www.sfu.ca/~ssurjano/rastr.html"""
        self._name = 'Rastrigin 1D'
        scan_c = kwargs.get('scan_constraint', False) # if moving the threshold

        dim = 1
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -5, torch.ones(dim) * 5
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        
        self.objective = lambda x: -Rastrigin(dim=1)(x)
        # self.c_func1 = lambda x: -(x+3)**2 + 0.64  # |x - -2| < 0.5
        # self.c_func1 = lambda x: -torch.abs(x+3.1)**(1/2) + .8 **(1/2) 
        self.c_func1 = lambda x: torch.abs(x+.7)**(1/2) - 2 **(1/2) 
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func_list = [self.c_func1_scbo]
        self.x_tensor = self._generate_x_tensor(dim=1, num=self._num_pts).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1]

        # feasible region identification
        self.constraint_confidence_list = [0.5]
        if scan_c:
            self.c_portion = kwargs.get('constrained_portion', .1) # portion of feasible region
            self.constraint_threshold_list = [np.quantile(self.c_tensor1, 1 - self.c_portion)] 
        else:
            self.constraint_threshold_list = [0]
            self.c_portion = (sum(self.c_tensor1 > 0) / self.c_tensor1.size(0)).detach().item()
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
        
    def ackley_5D_2C(self, scbo_format:bool=False) -> List[tensor]:
        '''
        Note: due to earlier problem, it is actually 4d. but feeded to a 10 d function. and only on diagonal.
        '''
        self._name = 'Ackely'
        dim = 5
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -5, torch.ones(dim) * 3
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        # self.objective = lambda x: Ackley(dim=10)(x) # deliberately set to be 10, still work on dim specified
        self.objective = lambda x: Ackley(dim=dim)(x) / 20 # standardize to make it comparable.
        # self.c_func1 = lambda x: -torch.sum(x) # sum x <= 0
        self.c_func1 = lambda x: -torch.max(x**2)+3**2 # max x**2 <= 9
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        # self.c_func2 = lambda x: - torch.linalg.vector_norm(x-torch.ones(dim))  + 4 # norm x < 5
        self.c_func2 = lambda x:  (torch.linalg.vector_norm(x-torch.ones(dim)) - 5.5)**2 - 1.**2 # norm x < 4 or > 7
        self.c_func2_scbo = lambda x: -self.c_func2(x)
        self.c_func_list = [self.c_func1_scbo, self.c_func2_scbo]
        self.x_tensor = self._generate_x_tensor(dim=dim, num=self._num_pts, seed=2).to(device=device, dtype=dtype)
        self.x_tensor = unnormalize(self.x_tensor, (torch.zeros(dim).to(device=device, dtype=dtype), torch.ones(dim).to(device=device, dtype=dtype)))
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor2 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func2, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1, self.c_tensor2]
        self.constraint_threshold_list = [0, 0]
        self.constraint_confidence_list = [0.5, 0.5]
        self.feasible_filter = feasible_filter_gen(self.c_tensor_list, self.constraint_threshold_list)
        self.c_portion = (sum(self.feasible_filter) / self.feasible_filter.size(0)).detach().item()

        assert torch.any(self.feasible_filter)

        __feasible_y = torch.where(self.feasible_filter, self.y_tensor.squeeze(), float('-inf'))
        self.maximum = __feasible_y.max()
        self.max_arg = __feasible_y.argmax()

        if not scbo_format:
            return self.x_tensor_range, self.y_tensor, self.c_tensor_list
            # return self.x_tensor, self.y_tensor, self.c_tensor_list
        else:
            return self.x_tensor, self.objective, self.c_func_list

    def rosenbrock_5d(self, scbo_format:bool=False, standardization:bool=True) -> List[tensor]:
        self._name = 'Rosenbrock_3C'
        dim = 2
        self.dim = dim
        self.lb, self.ub = torch.ones(dim) * -3, torch.ones(dim) * 5
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        if standardization:
            self.objective = lambda x: -0.5 * Ackley(dim=dim)(x) # deliberately set to be 10, still work on dim = 1
            self.c_func1 = lambda x: -DixonPrice(dim=dim)(x)/(dim*100) + .1
            self.c_func2 = lambda x: -Levy(dim=dim)(x)/(dim*100) + .1
            self.c_func3 = lambda x: torch.linalg.vector_norm(x, dim=-1)**(1/2) - (1.3)**(1/2)
        else:
            self.objective = lambda x: Ackley(dim=dim)(x) # deliberately set to be 10, still work on dim = 1
            self.c_func1 = lambda x: -DixonPrice(dim=dim)(x)
            self.c_func2 = lambda x: -Levy(dim=dim)(x)
            self.c_func3 = lambda x: torch.linalg.vector_norm(x, dim=-1)**(1/2) - (1.3)**(1/2)
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func2_scbo = lambda x: -self.c_func2(x)
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

    def rosenbrock_4d(self, scbo_format:bool=False) -> List[tensor]:
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

    def water_converter_32d(self, scbo_format:bool=False) -> List[tensor]:
        '''
        All subreward > 87682.7047, actually 2c
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
        self.objective = lambda x: Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=-1) / (raw_factor/_cali_factor) - 13 * _cali_factor
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

    def water_converter_32d_neg(self, scbo_format=False) -> List[tensor]:
        '''
        All subreward < 92000
        '''
        self._name = "Water_Converter_16C_neg"
        self.dim = 32
        _cali_factor = 10
        raw_threshold = 96000
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
        self.objective = lambda x: Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=-1) / (raw_factor/_cali_factor) - 13 * _cali_factor
        self.c_tensor_list = [(-raw_rewards[:, c_idx].reshape([-1,1]) + raw_threshold)/raw_factor for c_idx in range(c_num)]
        self.c_func_list = [lambda x: -(-Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=c_idx+32) + raw_threshold)/raw_factor for c_idx in range(c_num)]
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

    def water_converter_32d_neg_3c(self, scbo_format=False) -> List[tensor]:
        '''
        All subreward < 92000
        '''
        self._name = "Water_Converter_16C_neg_3c"
        self.dim = 32
        _cali_factor = 10
        raw_threshold = 96000
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
        self.objective = lambda x: Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=-1) / (raw_factor/_cali_factor) - 13 *  _cali_factor
        self.c_tensor_3 = -torch.linalg.norm(self.x_tensor_range, dim=-1)**(1/4) + 2000**(1/4)
        self.c_func_3 = lambda x: torch.linalg.norm(unnormalize(x, (self.lb, self.ub)), dim=-1)**(1/4) - 2000**(1/4)
        self.c_tensor_list = [(-raw_rewards[:, c_idx].reshape([-1,1]) + raw_threshold)/raw_factor for c_idx in range(c_num)]
        self.c_tensor_list.append(self.c_tensor_3.reshape([-1,1]))
        self.c_func_list = [lambda x: -(-Constrained_Data_Factory.nearest_approx(data, unnormalize(x, (self.lb, self.ub)), 32, reward_idx=c_idx+32) + raw_threshold)/raw_factor for c_idx in range(c_num)]
        self.c_func_list.append(self.c_func_3)
        self.constraint_threshold_list = [0 for _ in range(c_num+1)]
        self.constraint_confidence_list = [0.5 for _ in range(c_num+1)]

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


    def RE2_4D_3C(self, scbo_format:bool=False,):
        """An easy-to-use real-world multi-objective optimization problem suite - ScienceDirect
            [https://www.sciencedirect.com/science/article/pii/S1568494620300181#appSB]
            Pressure Vessel Design Problem.
        """
        self._name = 'vessel_4D_3C'

        dim = 4
        self.dim = dim
        self.lb, self.ub = torch.tensor([1, 1, 10, 10]), torch.tensor([100, 100, 200, 240])
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        
        self.objective = lambda x: 0.6224*x[0]*x[2]*x[3] + 1.7781*x[0]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2]
        # self.c_func1 = lambda x: -(x+3)**2 + 0.64  # |x - -2| < 0.5
        self.c_func1 = lambda x: x[0] - 0.0193*x[2]
        self.c_func1_scbo = lambda x: -self.c_func1(x)
        self.c_func2 = lambda x: x[1] - 0.00954*x[2]
        self.c_func2_scbo = lambda x: -self.c_func1(x)
        self.c_func3 = lambda x: np.pi*x[2]**2*x[3] + 4/3*np.pi*x[2]**3-1296000
        self.c_func3_scbo = lambda x: -self.c_func1(x)

        self.c_func_list = [self.c_func1_scbo, self.c_func2_scbo, self.c_func3_scbo]
        self.x_tensor = self._generate_x_tensor(dim=1, num=self._num_pts).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor1 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func1, self.x_tensor).unsqueeze(-1)
        self.c_tensor2 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func2, self.x_tensor).unsqueeze(-1)
        self.c_tensor3 = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.c_func3, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [self.c_tensor1, self.c_tensor2, self.c_tensor3]

        # feasible region identification
        self.constraint_confidence_list = [0.5, .5, .5]
        self.constraint_threshold_list = [0, 0, 0]
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
        return
    
    def RE2_3_5(self, scbo_format:bool=False,):
        """An easy-to-use real-world multi-objective optimization problem suite - ScienceDirect
            [https://www.sciencedirect.com/science/article/pii/S1568494620300181#appSB]
            Coil Compression Spring Design.
        """

        return
    
    def RE9_7D_8C(self, scbo_format:bool=False,):
        """An easy-to-use real-world multi-objective optimization problem suite - ScienceDirect
            [https://www.sciencedirect.com/science/article/pii/S1568494620300181#appSB]
            Note: the original supplementary material doens't match the description. We refer to github version:
            [https://github.com/ryojitanabe/reproblems/blob/master/reproblem_python_ver/reproblem.py]
            Car Cab Design.
        """
        self._name = 'car_cab_7D_8C'

        dim = 7
        c_num = 8
        self.dim = dim
        self.lb, self.ub = torch.tensor([.5, .45, .5, .5, .875, .4, .4]), torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1., 1.2])
        self.lb, self.ub =self.lb.to(device=device, dtype=dtype), self.ub.to(device=device, dtype=dtype)
        
        # stochastic variables
        np.random.seed(dim)
        x7 = lambda : 0.006 * (np.random.normal(0, 1)) + 0.345
        x8 = lambda : 0.006 * (np.random.normal(0, 1)) + 0.192
        x9 = lambda : 10 * (np.random.normal(0, 1)) + 0.0
        x10 = lambda : 10 * (np.random.normal(0, 1)) + 0.0

        # obj and constraints
        self.objective = lambda x: 1.98 + 4.9*x[0] + 6.67*x[1] + 6.98*x[2] + 4.01*x[3] + 1.78*x[4] + 0.00001*x[5] + 2.73*x[6]
        self._c_func_list = [None for _ in range(c_num)]
        self._c_func_list[0] = lambda x: 1 - (1.16 - 0.3717*x[1]*x[3] - 0.00931*x[1]*x9 - 0.484*x[2]*x8 + 0.01343*x[5]*x9)
        self._c_func_list[1] = lambda x: .32 - (.261 - .0159*x[0]*x[1] - .188*x[0]*x7 - .019*x[1]*x[6] 
                                                + .0144*x[2]*x[4] + .87570001*x[4]*x9 + 0.08045*x[5]*x8 
                                                + 0.00139*x7*x10 + .00001575*x9*x10)
        self._c_func_list[2] = lambda x: .32 - (0.214 + 0.00817 * x[4] - 0.131 * x[0] * x7 - 0.0704 * x[0] * x8
                                                + 0.03099 * x[1] * x[5] - 0.018 * x[1] * x[6] + 0.0208 * x[2] * x7 
                                                + 0.121 * x[2] * x8 - 0.00364 * x[4] * x[5] + 0.0007715 * x[4] * x9 
                                                - 0.0005354 * x[5] * x9 + 0.00121 * x7 * x10 + 0.00184 * x8 * x9 - 0.018 * x[1] * x[1])
        self._c_func_list[3] = lambda x: .32 - (0.74 - 0.61* x[1] - 0.163 * x[2] * x7 + 0.001232 * x[2] * x9 - 0.166 * x[6] * x8 + 0.227 * x[1] * x[1])
        
        self._c_func_list[4] = lambda x: 32 - (( 28.98 + 3.818 * x[2] - 4.2 * x[0] * x[1] + 0.0207 * x[4] * x9 + 6.63 * x[5] * x8 - 7.77 * x[6] * x7 + 0.32 * x8 * x9) 
                                               + (33.86 + 2.95 * x[2] + 0.1792 * x9 - 5.057 * x[0] * x[1] - 11 * x[1] * x7 - 0.0215 * x[4] * x9 - 9.98 * x[6] * x7 + 22 * x7 * x8) 
                                               + (46.36 - 9.9 * x[1] - 12.9 * x[0] * x7 + 0.1107 * x[2] * x9) )/3
        self._c_func_list[5] = lambda x: 32 - (4.72 - 0.5 * x[3] - 0.19 * x[1] * x[2] - 0.0122 * x[3] * x9 + 0.009325 * x[5] * x9 + 0.000191 * x10 * x10)
        self._c_func_list[6] = lambda x: 4 - (10.58 - 0.674 * x[0] * x[1] - 1.95  * x[1] * x7  + 0.02054  * x[2] * x9 - 0.0198  * x[3] * x9  + 0.028  * x[5] * x9)
        self._c_func_list[7] = lambda x: 9.9 - (16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4] * x[5] + 0.0432 * x8 * x9 - 0.0556 * x8 * x10 - 0.000786 * x10 * x10)
     
        
        self.c_func_list = [lambda x: -c_func(x)  for c_func in self._c_func_list]
        self.x_tensor = self._generate_x_tensor(dim=1, num=self._num_pts).to(device=device, dtype=dtype)
        self.x_tensor_range = unnormalize(self.x_tensor, (self.lb, self.ub))
        self.y_tensor = Constrained_Data_Factory.evaluate_func(self.lb, self.ub, self.objective, self.x_tensor).unsqueeze(-1)
        self.c_tensor_list = [Constrained_Data_Factory.evaluate_func(self.lb, self.ub, c_func, self.x_tensor).unsqueeze(-1) for c_func in self._c_func_list]

        # feasible region identification
        self.constraint_confidence_list = [0.5 for _ in range(c_num)]
        self.constraint_threshold_list = [0 for _ in range(c_num)]
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
        return

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
        if hasattr(self, 'c_portion'):
            _fig_dir = f"./res/illustration/{self._venue}_{self._name}_P{self.c_portion:.0%}"

        else:
            _fig_dir = f"./res/illustration/{self._venue}_{self._name}"
        plt.savefig(_fig_dir)
        plt.close()


