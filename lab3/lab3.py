import scipy.special as sp
import numpy as np

class BayesianPCA(object):
    
    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        """
        self.d = d # number of dimensions
        self.N = N # number of data points
        
        # Hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        # Variational parameters
        self.means_z = np.random.randn(d, N) # called x in bishop99
        self.sigma_z = np.random.randn(d, d)
        self.mean_mu = np.random.randn(d, 1)
        self.sigma_mu = np.random.randn(d, d)
        self.means_w = np.random.randn(d, d)
        self.sigma_w = np.random.randn(d, d)
        self.a_alpha_tilde = np.abs(np.random.randn(1))
        self.bs_alpha_tilde = np.abs(np.random.randn(d, 1))
        self.a_tau_tilde = np.abs(np.random.randn(1))
        self.b_tau_tilde = np.abs(np.random.randn(1))
    
    def __update_z(self, X):
        """
        Q(Z) = prod_n N(z_n|m_z,sigma_z)
            where m_z = <tau>sigma_z|<tau><W^T>(x_n-<mu>)
                  sigma_x = (I+<tau><W^T W>)^-1

        """ 
        #Expectation of tau
        exp_tau = self.a_tau_tilde / self.b_tau_tilde
        exp_WT = self.means_w.T
        exp_mu = self.mean_mu
        exp_WTW = self.means_w.T*self.means_w

        #update sigma_z
        self.sigma_z = (np.identity(self.d)+exp_tau*exp_WTW)**(-1)
        print self.sigma_z.shape


        pass
    
    def __update_mu(self):
        pass
    
    def __update_w(self, X):
        pass
    
    def __update_alpha(self):
        pass

    def __update_tau(self, X):
        pass

    def L(self, X):
        L = 0.0
        return L
    
    def fit(self, X):
        self.__update_z(X)

X = np.random.randn(4,2)
vPca = BayesianPCA(4,2)
vPca.fit(X)