import scipy.special as sp
import numpy as np

DEBUG = 1

class BayesianPCA(object):
    
    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        """
        self.d = d # number of dimensions
        self.N = N # number of data points
        self.q = d - 1 # number of latent space dimensionality
        
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
        # initialize latent variable z
        self.data = np.random.randn(d, N)
    
    def __update_z(self, X):
        """
        Q(Z) = prod_n N(z_n|m_z,sigma_z)
            where m_z = <tau>*sigma_z|<tau>*<W^T>*(x_n-<mu>)
                  sigma_x = (I+<tau>*<W^T*W>)^-1

        """
        #Expectation of tau
        exp_tau = self.a_tau_tilde / self.b_tau_tilde
        exp_WT = self.means_w.T
        exp_mu = self.mean_mu
        exp_WTW = self.means_w.T*self.means_w

        #update sigma_z
        self.sigma_z = (np.identity(self.d)+exp_tau*exp_WTW)**(-1)
        #update m_z
        for i in range(self.N):
            self.means_z[:,i] = (np.dot(exp_tau * self.sigma_z * exp_WT, (X[:,i].reshape(self.d,1) - exp_mu))).reshape(self.d)
            # generation of new latent data
            self.data[:,i] = np.random.multivariate_normal(self.means_z[:,i], self.sigma_z, 1)
        if DEBUG:
            print "============= UPDATE_Z =============="
            print "self.means_z\n", self.means_z
            print "self.data\n", self.data
    
    def __update_mu(self):
        """
        Q(mu) = N(mu|m_mu,sigma_mu)
            where m_mu = <tau>*sigma_mu*sum_n(x_n-<W><z_n>)
                  sigma_mu = (beta+N<tau>)^-1 * I
        """
        exp_tau = self.a_tau_tilde / self.b_tau_tilde
        #update sigma_mu
        self.sigma_mu = (self.beta + self.N * exp_tau)**(-1) * np.identity(self.d)
        #update mean_mu
        self.mean_mu = np.diag(exp_tau * self.sigma_mu * np.sum(X - np.dot(self.means_w, self.means_z), axis = 1))
        if DEBUG:
            print "============= UPDATE_MU =============="
            print "self.sigma_mu\n", self.sigma_mu
            print "self.sigma_mu\n", self.mean_mu
    
    def __update_w(self, X):
        a_diag = np.diag(np.zeros((self.d,self.d)) + self.a_alpha / self.b_alpha)
        exp_tau = self.a_tau_tilde / self.b_tau_tilde
        self.sigma_w = np.sum(np.dot(self.means_z, self.means_z.T), axis = 0)

        # TODO means_w

        # for k in xrange(self.d):
        #     sum_ = 0
        #     for n in xrange(self.N):
        #         sum_ += self.means_z[n] * (X[k,n] - self.mean_mu[k])
        #     print sum_
        #     print self.sigma_w
        #     self.means_w[k] = self.sigma_w * sum_
        # print self.means_w

        if DEBUG:
            print "============= UPDATE_W =============="
            print "self.sigma_w\n", self.sigma_w
            print "self.sigma_w\n", self.means_w

    
    def __update_alpha(self):
        self.a_alpha_tilde = self.a_alpha + self.d / 2
        self.a_tau_tilde = self.a_tau + .5 * (self.N * self.d)
        if DEBUG:
            print "============= UPDATE_alpha =============="
            print "self.a_alpha_tilde\n", self.a_alpha_tilde
            print "self.a_tau_tilde\n", self.a_tau_tilde

    def __update_tau(self, X):
        # TODO
        for i in xrange(self.d):
            self.bs_alpha_tilde[i] = 0
        if DEBUG:
            print "============= UPDATE_tau =============="
            print "self.bs_alpha_tilde\n", self.bs_alpha_tilde
            print "self.b_tau_tilde\n", self.b_tau_tilde

    def L(self, X):
        L = 0.0
        return L
    
    def fit(self, X):
        self.__update_z(X)
        self.__update_mu()
        self.__update_w(X)
        self.__update_alpha()
        self.__update_tau(X)

X = np.random.randn(4,2)
vPca = BayesianPCA(4,2)
vPca.fit(X)