import scipy.special as sp
import numpy as np
import pylab as P
import sys
import cPickle, gzip

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
        # expectation of tau
        self.exp_tau = self.a_tau_tilde / self.b_tau_tilde

    def __checkSizes(self):
        if(self.means_z.shape != (self.d, self.N)):
            print "ERROR self.means_z.shape"
            sys.exit()
        if(self.sigma_z.shape != (self.d, self.d)):
            print "ERROR self.sigma_z"
            sys.exit()
        if(self.mean_mu.shape != (self.d, 1)):
            print "ERROR self.mean_mu"
            sys.exit()
        if(self.sigma_mu.shape != (self.d, self.d)):
            print "ERROR elf.sigma_mu"
            sys.exit()
        if(self.means_w.shape != (self.d, self.d)):
            print "ERROR self.means_w"
            sys.exit()
        if(self.sigma_w.shape != (self.d, self.d)):
            print "ERROR self.sigma_w"
            sys.exit()
        if(self.bs_alpha_tilde.shape != (self.d, 1)):
            print "ERROR self.bs_alpha_tilde"
            sys.exit()
    
    def __update_z(self, X):
        """
        Q(Z) = prod_n N(z_n|m_z,sigma_z)
            where m_z = <tau>*sigma_z|<tau>*<W^T>*(x_n-<mu>)
                  sigma_x = (I+<tau>*<W^T*W>)^-1

        """    
        #update sigma_z
        self.sigma_z = np.linalg.inv(np.identity(self.d) + np.multiply(self.exp_tau, np.trace(self.sigma_w) + np.dot(self.means_w.T, self.means_w)))
        # update mean_z
        self.means_z = self.exp_tau*np.dot(np.dot(self.sigma_z, self.means_w.T) , (X - self.mean_mu))
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
        #update sigma_mu
        self.sigma_mu = (self.beta + self.N * self.exp_tau)**(-1) * np.identity(self.d)
        #update mean_mu
        self.mean_mu = self.exp_tau * np.dot(self.sigma_mu,np.sum(X - np.dot(self.means_w, self.means_z), axis=1)).reshape((self.d,1))

        if DEBUG:
            print "============= UPDATE_MU =============="
            print "self.sigma_mu\n", self.sigma_mu
            print "self.sigma_mu\n", self.mean_mu
    
    def __update_w(self, X):
        # sigma of w
        a_diag = np.diagflat(self.a_alpha_tilde/self.bs_alpha_tilde)
        sum_ = np.zeros((self.d, self.d))
        for n in xrange(self.N):
            sum_ += self.sigma_z + np.dot(self.means_z[:,n].reshape((self.d,1)), self.means_z[:,n].T.reshape((1,self.d)))
        self.sigma_w = np.linalg.inv(a_diag + self.exp_tau * sum_)
        # means of w
        self_means_w = np.dot(self.exp_tau*self.sigma_w, np.dot(self.means_z, (X - self.mean_mu).T)).T
        # debug prints
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
        # debug prints
        if DEBUG:
            print "============= UPDATE_alpha =============="
            print "self.a_alpha_tilde\n", self.a_alpha_tilde
            print "self.bs_alpha_tilde\n", self.bs_alpha_tilde

    def __update_tau(self, X):
        # update a_tau_tilde
        self.a_tau_tilde = self.a_tau + (self.N * self.d) / 2
        sum_ = 0
        # update b_tau_tilde
        for n in xrange(self.N):
            sum_ += np.dot(X.T[n] , X.T[n]) + np.dot(self.means_z.T[n], self.means_z.T[n])
            + np.trace((self.means_w.T*self.means_w) * (self.sigma_z + np.dot(self.means_z, self.means_z.T)))
            + 2 * np.dot(np.dot(self.mean_mu.T, self.means_w), self.means_z.T[n])
            -2 * np.dot(X.T[n], self.mean_mu)
            -2 * np.dot(np.dot(X.T[n], self.means_w), self.means_z[:,n])
        self.b_tau_tilde = self.b_tau + sum_ / 2
        # debug prints
        if DEBUG:
            print "============= UPDATE_tau =============="
            print "self.bs_alpha_tilde\n", self.a_tau_tilde
            print "self.b_tau_tilde\n", self.b_tau_tilde

    def L(self, X):
        L = 0.0
        return L

    def CheckFittedModel(self, X):
        print "Checking if the model is well fitted..."
        print "Matrices must be identical:"
        print X
        print np.dot(self.means_w,self.means_z) + self.mean_mu
    
    def fit(self, X):
        iterations = 10000
        print "fitting the model..."
        for x in xrange(iterations):
            if (x%(iterations/10.0)==0.0):
                print ".",
            vPca.__checkSizes()
            self.__update_z(X)
            self.__update_mu(X)
            self.__update_w(X)
            self.__update_alpha()
            self.__update_tau(X)
        print "\n",iterations, "iterations done"
    def _blob(self,x,y,area,colour):
        """
        Draws a square-shaped blob with the given area (< 1) at
        the given coordinates.
        Source: http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
        """
        hs = np.sqrt(area) / 2
        xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
        P.fill(xcorners, ycorners, colour, edgecolor=colour)

    def hinton(self, maxWeight=None):
        """
        Draws a Hinton diagram for visualizing a weight matrix. 
        Temporarily disables matplotlib interactive mode if it is on, 
        otherwise this takes forever.
        Source: http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
        """
        reenable = False
        if P.isinteractive():
            P.ioff()
        P.clf()
        height, width = self.sigma_w.shape
        if not maxWeight:
            maxWeight = 2**np.ceil(np.log(np.max(np.abs(self.sigma_w)))/np.log(2))

        P.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
        P.axis('off')
        P.axis('equal')
        for x in xrange(width):
            for y in xrange(height):
                _x = x+1
                _y = y+1
                w = self.sigma_w[y,x]
                if w > 0:
                    self._blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
                elif w < 0:
                    self._blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
        if reenable:
            P.ion()
        P.show()


mean = np.zeros(10)
cov = np.diag([5,4,3,2,1,1,1,1,1,1])
X = np.random.multivariate_normal(mean,cov,100).T
vPca = BayesianPCA(10,100)
vPca.fit(X)
vPca.CheckFittedModel(X)
# http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
vPca.hinton()

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()