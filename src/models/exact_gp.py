'''
Model for exact GP
Shall support multiple acquisition functions:
UCB, TS, (q)EI.
'''
import gpytorch

class ExactGPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor, low_dim=True):
            '''
            Exact GP:
            Leave placeholder for gp_feature_extractor
            '''
            super(ExactGPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
            # self.mean_module = gpytorch.means.ZeroMean()
            if low_dim:
                # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                self.covar_module = gpytorch.kernels.ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    gpytorch.kernels.MaternKernel(
                    nu=2.5, ard_num_dims=train_x.size(-1), 
                    lengthscale_constraint=gpytorch.constraints.Interval(0.005, 4.0)
                ))
            else:
                self.covar_module = gpytorch.kernels.LinearKernel(num_dims=train_x.size(-1))
            try: # gpytorch 1.6.0 support
                self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
            except Exception: # gpytorch 1.9.1
                self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1)),
            #     num_dims=train_x.size(-1), grid_size=1000)
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1),
            #     outputscale_constraint=gpytorch.constraints.Interval(0.7,1.0)),
            #     num_dims=train_x.size(-1), grid_size=100)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            self.projected_x = x
            # self.projected_x = self.scale_to_bounds(x)  # Make the values "nice"

            mean_x = self.mean_module(self.projected_x)
            covar_x = self.covar_module(self.projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)