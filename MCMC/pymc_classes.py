import numpy as np
from scipy.integrate import odeint
from scipy.linalg import lstsq
from scipy.optimize import approx_fprime

import theano
import theano.tensor as T



def Phi(phis, wavelengths, lambda_C=400):
    return sum(par * ((lambda_C - wavelengths) / 100) ** i for i, par in enumerate(phis))
   
# K must be  w x n x n matrix where w is number of wavelenghts
def simulate(times, K, eps, q_tot, c0, V, I_source):

    const = np.log(10)

    def dc_dt(c, t):
        c_eps = c[:, None] * eps  # hadamard product
        c_dot_eps = c_eps.sum(axis=0)

        q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
                             (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

        # batched dot product for each wavelength
        product = np.matmul(K, q.T[..., None]).squeeze()  # w x n x 1

        return q_tot / V * (product.sum(0) - (product[0] + product[-1]) / 2)

    return odeint(dc_dt, c0, times)

# K must be just n x n matrix
def simulate_no_wl_depend(times, K, eps, q_tot, c0, V, I_source):
    
    const = np.log(10)

    def dc_dt(c, t):
        c_eps = c[:, None] * eps  # hadamard product
        c_dot_eps = c_eps.sum(axis=0)

        q = c_eps * np.where(c_dot_eps <= 0.001, const - c_dot_eps * const * const / 2,
                             (1 - np.exp(-c_dot_eps * const)) / c_dot_eps) * I_source

        integral = q.sum(axis=1) - (q[:, 0] + q[:, -1]) / 2  # trapz integration

        return q_tot / V * K.dot(integral[:, None]).squeeze()

    return odeint(dc_dt, c0, times)

def log_likelihood(params, D, times, wavelengths, eps_est, q_tot, c0, V, I_source, n_MCR_iter=5, no_wl_dependence=True):

    # optimize spectra for curent C params by MCR-ALS style
    if no_wl_dependence:
        phi_ZE = params[0]
        phi_EZ = params[1]
    else:
        phi = Phi([params[0], 0, params[1]], wavelengths, lambda_C=400)
        
        # add potential
        if any(phi < 0) or any(phi > 1):
            return -1.7976931348623157e+10

    _0 = 0 if no_wl_dependence else np.zeros_like(wavelengths)

#     K = np.asarray([[-phi, _0],
#                     [+phi, _0]])

    K = np.asarray([[-phi_ZE,  phi_EZ],
                    [+phi_ZE, -phi_EZ]])

    if not no_wl_dependence:
        K = np.transpose(K, (2, 0, 1))
        
#     C = np.zeros((times.shape[0] * 2, wavelengths.shape[0]))

    eps_opt = eps_est.copy()

    for i in range(n_MCR_iter):
        # calc C
#         eps_opt[0] = eps_est[0].copy()
        
#         C1 = simulate_no_wl_depend(times, K, eps_opt, q_tot, c0, V, I_source) if no_wl_dependence else simulate(times, K, eps_opt, q_tot, c0, V, I_source)
        
        C1 = simulate_no_wl_depend(times, K, eps_opt, q_tot, [c0, 0], V, I_source)
        C2 = simulate_no_wl_depend(times, K, eps_opt, q_tot, [0, c0], V, I_source) 
        
        C = np.vstack((C1, C2))

        # calc ST by lstsq
        eps_opt = lstsq(C, D)[0]

        # apply non-negative contraints on spectra
        eps_opt *= (eps_opt > 0)
        
#         eps_opt[0] = eps_est[0].copy()

    #         self.calls.append([params[0], self.eps_est])
    D_sim = C.dot(eps_opt)
    residuals = D - D_sim

    # calculate the log of gaussian likelihood
    N = 1  # D.size
    #         LL = -0.5*N*np.log(2*np.pi*sigma**2) - (0.5/sigma**2) * (residuals**2).sum()
    
    sigma = 0.01

    LL = - (0.5 / sigma ** 2) * (residuals ** 2).sum()
    return LL, eps_opt




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

    def __init__(self, log_like, D, times, wavelengths, eps_est, I_source, q_tot, V, c0):
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
#         self.calls = []
        self.log_like = log_like
        self.logpgrad = LogLikeGrad(self.log_like, self.D, self.times, self.wavelengths, self.eps_est, 
                              self.I_source, self.q_tot, self.V, self.c0)

        
    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        params, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl, _ = self.log_like(params, self.D, self.times, self.wavelengths, self.eps_est, 
                              self.q_tot, self.c0, self.V, self.I_source, n_MCR_iter=5)

        outputs[0][0] = np.array(logl) # output the log-likelihood
    
    def _log_like(self, params):
        return self.log_like(params, self.D, self.times, self.wavelengths, self.eps_est, 
                              self.q_tot, self.c0, self.V, self.I_source, n_MCR_iter=5)
        
    
    def _grads(self, params):
        return approx_fprime(params, self._log_like, 1e-4)
        
    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values 
        theta, = inputs  # our parameters 
        return [g[0]*self.logpgrad(theta)]
        

        
class LogLikeGrad(T.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [T.dvector]
    otypes = [T.dvector]

    def __init__(self, log_like, D, times, wavelengths, eps_est, I_source, q_tot, V, c0):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.log_like = log_like
        self.D = D.copy()
        self.times = times.copy()
        self.wavelengths = wavelengths.copy()
        self.I_source = I_source.copy()
        self.q_tot = q_tot
        self.V = V
        self.c0 = c0  # initial conditions
        self.eps_est = eps_est  # estimate of spectra == ST

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(params):
            ll, _ = self.log_like(params, self.D, self.times, self.wavelengths, self.eps_est, self.q_tot, self.c0, self.V, self.I_source, n_MCR_iter=5)
            return ll 

        # calculate gradients
        grads = approx_fprime(theta, lnlike, 1e-4)

        outputs[0][0] = grads