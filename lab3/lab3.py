import scipy.special as sp
import numpy as np

DEBUG = 0

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
        # initialize latent variable z
        self.data = np.random.randn(d, N)

    def __checkSizes(self):
        if(self.means_z.shape != (self.d, self.N)):
            print "ERROR.shape == ()"
        if(self.sigma_z.shape != (self.d, self.d)):
            print "ERRORself.sigma_z"
        if(self.mean_mu.shape != (self.d, 1)):
            print "ERRORself.mean_mu"
        if(self.sigma_mu.shape != (self.d, self.d)):
            print "ERRORelf.sigma_mu"
        if(self.means_w.shape != (self.d, self.d)):
            print "ERRORself.means_w"
        if(self.sigma_w.shape != (self.d, self.d)):
            print "ERRORself.sigma_w"
        if(self.bs_alpha_tilde.shape != (self.d, 1)):
            print "ERROR_bs_alpha_tilde"
    
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
        exp_WTW = np.trace(self.sigma_w) + np.dot(self.means_w.T, self.means_w)
        
        #update sigma_z
        self.sigma_z = np.linalg.inv(np.identity(self.d) + np.multiply(exp_tau, exp_WTW))
        # TODO make mean_z
        self.means_z = exp_tau*np.dot(np.dot(self.sigma_z, self.means_w.T) , (X - self.mean_mu))
        
        if DEBUG:
            print "============= UPDATE_Z =============="
            print "self.means_z\n", self.means_z
            print "self.sigma_z\n", self.sigma_z
    
    def __update_mu(self, X):
        """
        Q(mu) = N(mu|m_mu,sigma_mu)
            where m_mu = <tau>*sigma_mu*sum_n(x_n-<W><z_n>)
                  sigma_mu = (beta+N<tau>)^-1 * I
        """
        exp_tau = self.a_tau_tilde / self.b_tau_tilde
        #update sigma_mu
        self.sigma_mu = (self.beta + self.N * exp_tau)**(-1) * np.identity(self.d)
        #update mean_mu
        self.mean_mu = exp_tau * np.dot(self.sigma_mu,np.sum(X - np.dot(self.means_w, self.means_z), axis=1)).reshape((self.d,1))

        if DEBUG:
            print "============= UPDATE_MU =============="
            print "self.sigma_mu\n", self.sigma_mu
            print "self.sigma_mu\n", self.mean_mu
    
    def __update_w(self, X):
        # TODO fix this
        # # sigma of w
        # exp_tau = self.a_tau_tilde / self.b_tau_tilde
        # a_diag = np.identity(self.d) * (self.a_alpha_tilde / self.b_tau_tilde)
        # self.sigma_w =  np.linalg.inv(a_diag + exp_tau * np.sum(self.sigma_z +  np.dot(self.means_z, self.means_z.T), axis=1))
        # # means of w
        # sum_ = np.zeros(self.d)
        # for n in xrange(self.N):
        #     sum_ += self.means_z[:,n] * (X[:,n] - self.mean_mu)
        # self.means_w = exp_tau * self.sigma_w * sum_

        # stathis
        self.sigma_w = np.linalg.inv(np.diagflat(self.a_alpha_tilde/self.bs_alpha_tilde) + (self.a_tau_tilde/self.b_tau_tilde)*(self.N*self.sigma_z + np.dot(self.means_z,self.means_z.T)) )
        self_means_w = np.dot((self.a_tau_tilde/self.b_tau_tilde)*self.sigma_w, np.dot(self.means_z, (X - self.mean_mu).T)).T
        
        if DEBUG:
            print "============= UPDATE_W =============="
            print "self.sigma_w\n", self.sigma_w
            print "self.means_w\n", self.means_w

    
    def __update_alpha(self):
        # a_alpha_tilted
        self.a_alpha_tilde = self.a_alpha + self.d / 2
        # bs_alpha_tilde
        for i in xrange(self.d):
            self.bs_alpha_tilde[i] = self.b_alpha + np.dot(self.means_w[i], self.means_w.T[i])/2

        if DEBUG:
            print "============= UPDATE_alpha =============="
            print "self.a_alpha_tilde\n", self.a_alpha_tilde
            print "self.bs_alpha_tilde\n", self.bs_alpha_tilde

    def __update_tau(self, X):
        self.a_tau_tilde = self.a_tau + (self.N * self.d) / 2
        sum_ = 0
        for n in xrange(self.N):
            sum_ += np.dot(X.T[n] , X.T[n]) + np.dot(self.means_z.T[n], self.means_z.T[n])
            + np.trace((self.means_w.T*self.means_w) * (self.sigma_z + np.dot(self.means_z, self.means_z.T)))
            + 2 * np.dot(np.dot(self.mean_mu.T, self.means_w), self.means_z.T[n])
            -2 * np.dot(X.T[n], self.mean_mu)
            -2 * np.dot(np.dot(X.T[n], self.means_w), self.means_z[:,n])

        self.b_tau_tilde = self.b_tau + sum_ / 2
        if DEBUG:
            print "============= UPDATE_tau =============="
            print "self.bs_alpha_tilde\n", self.a_tau_tilde
            print "self.b_tau_tilde\n", self.b_tau_tilde

    def L(self, X):
        L = 0.0
        return L
    
    def fit(self, X):
        iterations = 100000
        for x in xrange(iterations):
            print x
            vPca.__checkSizes()
            self.__update_z(X)
            self.__update_mu(X)
            self.__update_w(X)
            self.__update_alpha()
            self.__update_tau(X)
        print X
        print np.dot(self.means_w,self.means_z) + self.mean_mu

X = np.random.randn(4,2)
vPca = BayesianPCA(4,2)
vPca.fit(X)