from scipy.integrate import odeint
from scipy.linalg import lstsq
import numpy as np

import theano
import theano.tensor as T




# define a theano Op for our likelihood function
class LogLike(T.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [T.dvector] # expects a vector of parameter values when called
    otypes = [T.dscalar] # outputs a single scalar value (the log likelihood)
    
    # imputs phi, sigma

    def __init__(self, D, times, wavelengths, eps_est, I_source, q_tot, V, c0):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.D = D.copy()
        self.times = times.copy()
        self.wavelengths = wavelengths.copy()
        self.I_source = I_source.copy()
        self.q_tot = q_tot
        self.V = V
        self.c0 = c0  # initial conditions
        self.eps_est = eps_est  # estimate of spectra == ST
        
        self._0 = np.zeros_like(self.wavelengths)
#         self.calls = []

        
    def Phi(self, phis, lambda_C=400):
#         assert isinstance(phis, (list, np.ndarray))
        return sum(par * ((lambda_C - self.wavelengths) / 100) ** i for i, par in enumerate(phis))
   
    
    def simulate(self, times, K, eps, q_tot, c0, V, I_source):
        
        const = np.log(10)
        
        def dc_dt(c, t):
            c_eps = c[:, None] * eps  # hadamard product
            c_dot_eps = c_eps.sum(axis=0)

            q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
                                             (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

            product = np.matmul(K, q.T[..., None]).squeeze()  # w x n x 1

            return q_tot / V * (product.sum(0) - (product[0] + product[-1]) / 2)
        
        return odeint(dc_dt, c0, times)
    
        
    def log_likelihood(self, params, n_MCR_iter=10):
        
        # optimize spectra for curent C params by MCR-ALS style
        
        phi = self.Phi([params[0]], lambda_C=400)
        sigma = params[1]
        
        K = np.asarray([[-phi,  self._0],
                        [+phi,  self._0]])
        
        K = np.transpose(K, (2, 0, 1))
        C = np.zeros((self.times.shape[0], K.shape[0]))
        
        for i in range(n_MCR_iter):
            # calc C
            C = self.simulate(self.times, K, self.eps_est, self.q_tot, self.c0, self.V, self.I_source)
            
            # calc ST by lstsq
            self.eps_est = lstsq(C, self.D)[0]
            
            # apply non-negative contraints on spectra
            self.eps_est *= (self.eps_est > 0)
            
#         self.calls.append([params[0], self.eps_est])
        D_sim = C.dot(self.eps_est)
        residuals = self.D - D_sim
        
        # calculate the log of gaussian likelihood
        N = 1 #D.size
#         LL = -0.5*N*np.log(2*np.pi*sigma**2) - (0.5/sigma**2) * (residuals**2).sum()
        LL =  - (0.5/sigma**2) * (residuals**2).sum()

        
        return LL
        

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        params, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.log_likelihood(params)

        outputs[0][0] = np.array(logl) # output the log-likelihood