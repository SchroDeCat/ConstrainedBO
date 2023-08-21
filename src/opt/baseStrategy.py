'''
Abstract classes for LLNL usage.
'''

from abc import ABCMeta, abstractmethod
import os
import time
import pickle


class BaseStrategy(object):
    """
    Abstract class for a base ML predictor

    Parameters
    ----------- 
    name: string
        Name of the method.
    
    Attributes
    ----------
    name: str
        Name of method.

    Methods
    -------
    fit(X, y, **kwargs)
        Fits parameters of model to data.
    predict(X, **kwargs)
        Use model to predict ddG values using X features.
    load_model(X, y, **kwargs)
        Load model parameters into model.
    save_model(path)
        Save model parameters to a path.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):  # , full_name):
        self.name = name
        self.recording = False

    def __str__(self):
        return self.name

    @abstractmethod
    def save(self, path):
        timestamp = time.strftime("%D_%s").replace('/','')
        if os.path.isfile(path):
            os.rename(path, path + f'_{timestamp}')
        with open(path, 'wb') as f: 
            pickle.dump(self, f)
    
    @abstractmethod
    def load(self, path):
        with open(path, 'rb') as f: 
            self = pickle.load(f)

class GenerativeStrategy(BaseStrategy):
    """
    Abstract class for a base ML predictor

    Parameters
    ----------- 
    name: string
        Name of the method.
    
    Attributes
    ----------
    name: str
        Name of method.

    Methods
    -------
    fit(X, y, **kwargs)
        Fits parameters of model to data.
    predict(X, **kwargs)
        Use model to predict ddG values using X features.
    load_model(X, y, **kwargs)
        Load model parameters into model.
    (path)
        Save model parameters to a path.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, iters=20):  # , full_name):
        self.name = name
        self.recording = False
        self.iters = iters

    def __str__(self):
        return self.name
    
class MenuStrategy(BaseStrategy):
    """
    Abstract class for a base ML predictor
    
    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    model: `GPBase`
        underlying model that follows specification for basic model.
    train: bool
        Boolean whether we should train underlying model on some data before running selection.
    iters: int
        number of iterations to train underlying model for during intialization. Only used if train is `self.train = True`.
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, model=None, train=False, iters=20):  # , full_name):
        self.model = model
        self.name = name
        self.train = train
        self.iters = iters

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self,X,y,**kwargs):
        """
        Fit underlying model to data X using ddG values y

        Args:
            :param torch.Tensor X: The training inputs.
            :param torch.Tensor y: The training targets.

        """
        self.model.fit(X,y,**kwargs)


    @abstractmethod
    def select(self, X, i, k, **kwargs):
        """
        Select batch of `k` sequences from `X`. 

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor i: N shaped vector of encoded fidelities. (TODO: this should be a kwarg)
            :param int k: number of sequences to select

        Return:
            :return: list of indices that were selected. 
            :rtype: list
            ---- **optional**: these return values can be None ----
            :return:  mean value of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return: stddev of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return:  instantiation of torch.distribution joint_mvn covariance of posterior. 
            :rtype: `torch.Tensor`
        """
        pass

    @abstractmethod
    def evaluate(self, X, i):
        """
        Evaluate internal model at X sequences with i fidliety. 

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor i: N shaped vector of encoded fidelities. (TODO: this should be a kwarg)

        """
        pass

    @abstractmethod
    def score(self,X, joint_mvn_mean, joint_mvn_covar):
        """
        Internally score seqs in X based on `joint_mvn_mean` and `joint_mvn_covar`
        This method allows for flexible scoring rule, like adding penalties for
        mutational distance, a specific mutation, etc.

        Args:
            :param torch.Tensor X: NxD matrix where N is the number of seqs and D is # of features.
            :param torch.Tensor joint_mvn_mean: N shaped vector of mean value from predicitve posterior
            :param torch.Tensor joint_mvn_covar:   NxD covariance.  NOTE: not sure if this should lazy version provided by Gpytorch
            or the evaluated  matrix as tensor. For now I put tensor.
         Return:
            :return: scores for each indv. 
            :rtype: list
            ---- **optional**: these return values can be None ----
            :return:  mean value of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            :return: stddev of predictive posterior distribution obtained during selection.
            :rtype: `torch.Tensor`
            
        """
        pass

    @abstractmethod
    def update(self,X,y,**kwargs):
        """
        Update internal model at `X` seqs with `y` observations
        """
        self.model.resume_training(X,y,**kwargs)
