import numpy as np
from pylab import *

loopy = 1

class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
        
    def reset(self):
        # Incomming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        self.in_msgs[other] = msg
        if loopy:  
            for neighbour in self.neighbours:
                if neighbour is not other:
                    self.pending.add(neighbour)
        else:
            # Store the incomming message, replacing previous messages from the same node
            if len(self.in_msgs) == len(self.neighbours):
                for neighbour in self.neighbours:
                    self.pending.add(neighbour)
            if len(self.in_msgs) == len(self.neighbours) - 1:
                for neighbour in self.neighbours:
                    if neighbour not in self.in_msgs:
                        self.pending.add(neighbour)
                        break
        # print self.name
            
    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate observed an latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
         Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        """
        # TODO: compute marginal
        marginal = self.observed_state
        if len(self.in_msgs) == len(self.neighbours):
            for neighbour in self.neighbours:
                marginal = np.multiply(marginal, self.in_msgs[neighbour.name])
            if Z is None:
                Z = sum(marginal)
            marginal /= Z
        return marginal, Z
    
    def send_sp_msg(self, other):
        num_neighbours = len(self.neighbours)
        if num_neighbours == 1:
            other.receive_msg(self, self.observed_state)
        else:
            vectors = []
            num_of_incoming_messages = 0
            for neighbour in self.neighbours:
                if neighbour is not other and neighbour in self.in_msgs:
                    num_of_incoming_messages += 1
            if num_of_incoming_messages == num_neighbours - 1:
                message = np.ones(2)
                for neighbour in self.neighbours:
                    if neighbour is not other:
                        message = np.multiply(message, self.in_msgs[neighbour])
                        message = np.multiply(message, self.observed_state)
                other.receive_msg(self, message)

    def send_ms_msg(self, other):
        # http://www.cedar.buffalo.edu/~srihari/CSE574/Chap8/Ch8-GraphicalModelInference/Ch8.3.3-Max-SumAlg.pdf
        # TODO: implement Variable -> Factor message for max-sum
        num_neighbours = len(self.neighbours)
        if num_neighbours == 1:
            other.receive_msg(self, self.observed_state)
        else:
            vectors = []
            num_of_incoming_messages = 0
            for neighbour in self.neighbours:
                if neighbour is not other and neighbour in self.in_msgs:
                    num_of_incoming_messages += 1
            if num_of_incoming_messages == num_neighbours - 1:
                for neighbour in self.neighbours:
                    if neighbour is not other:
                        vectors.append(self.in_msgs[neighbour])
                message = np.add.reduce(vectors)
                message = np.multiply(message, self.observed_state)
                other.receive_msg(self,message)


                    

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f
        
    def send_sp_msg(self, other):
        # TODO: implement Factor -> Variable message for sum-product
        vectors = []
        if len(self.neighbours) == 1:
            other.receive_msg(self, self.f)
        else:            
            num_of_incoming_messages = 0
            num_neighbours = len(self.neighbours)
            for neighbour in self.neighbours:
                if neighbour is not other and neighbour in self.in_msgs:
                    num_of_incoming_messages += 1
            if num_of_incoming_messages == num_neighbours - 1:
                for neighbour in self.neighbours:
                    if neighbour is not other:
                        vectors.append(self.in_msgs[neighbour])
                tensor = np.multiply.reduce(np.ix_(*vectors))
                index = -1
                for i in range(num_neighbours):
                    if self.neighbours[i] is other:
                        index = i
                        break
                direction = filter(lambda i: not i == index, range(num_neighbours))
                message = np.tensordot(self.f, tensor, axes=(direction, range(tensor.ndim)))
                other.receive_msg(self, message)
                if other in self.pending:
                    self.pending.remove(other)
   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        vectors = []
        if len(self.neighbours) == 1:
            other.receive_msg(self, np.log(self.f))
        else:            
            num_of_incoming_messages = 0
            num_neighbours = len(self.neighbours)
            for neighbour in self.neighbours:
                if neighbour is not other and neighbour in self.in_msgs:
                    num_of_incoming_messages += 1
            if num_of_incoming_messages == num_neighbours - 1:
                for neighbour in self.neighbours:
                    if neighbour is not other:
                        vectors.append(self.in_msgs[neighbour])
                tensor = np.multiply.reduce(np.ix_(*vectors))
                index = -1
                for i in range(num_neighbours):
                    if self.neighbours[i] is other:
                        index = i
                        break
                tensor = np.add.reduce(np.ix_(*vectors))

                tile_factor = np.ones(num_neighbours)
                tile_factor[0] = other.num_states
                tiled_matrix = np.tile(tensor, tile_factor)

                res_tiled_matrix = tiled_matrix.reshape(self.f.shape)
                log_added = np.log(res_tiled_matrix) + np.log(res_tiled_matrix)
                direction = filter(lambda i: not i == index, range(num_neighbours))
                for dirct in direction:
                    maximum = np.maximum.reduce(log_added, dirct)
                other.receive_msg(self, maximum)
                if other in self.pending:
                    self.pending.remove(other)

class Factor_Graph(object):
    def __init__(self):
        # lists and dictionaries for variables and factors
        self.variable_list = {}
        self.factor_list = {}
        self.node_list = []
        self.ordered_node_list = []

    # 1.6
    def sum_product(self):
        pending_messages = []
        for node in self.ordered_node_list:
            for i in range(len(node.pending)):
                pending_messages.append((node, node.pending.pop()))
        for pm in pending_messages:
            pm[0].send_sp_msg(pm[1])
            print pm[0],"-->",pm[1]

        pending_reversed_messages = []
        for node in reversed(self.ordered_node_list):
            for i in range(len(node.pending)):
                pending_reversed_messages.append((node, node.pending.pop()))
        for pm in pending_reversed_messages:
            pm[0].send_sp_msg(pm[1])
            print pm[0],"-->",pm[1]

    # 2.1 Max-Sum
    def max_sum(self):
        loopy = 0
        pending_messages = []
        for node in self.ordered_node_list:
            # print node, len(node.pending)
            for i in range(len(node.pending)):
                pending_messages.append((node, node.pending.pop()))
        for pm in pending_messages:
            pm[0].send_ms_msg(pm[1])
            print pm[0],"-->",pm[1]

        pending_reversed_messages = []
        for node in reversed(self.ordered_node_list):
            # print node, len(node.pending)
            for i in range(len(node.pending)):
                pending_reversed_messages.append((node, node.pending.pop()))
        for pm in pending_reversed_messages:
            pm[0].send_ms_msg(pm[1])
            print pm[0],"-->",pm[1]

    # 3.2 Loopy Max-Sum
    def loopy_max_sum(self, max_iter):
        loopy = 1
        for x in xrange(max_iter):
            print "iteration: ", x , "/" , max_iter
            pending_messages = []
            for node in self.ordered_node_list:
                # print node, len(node.pending)
                for pending_node in node.pending:
                    pending_messages.append((node, pending_node))
            if len(pending_messages) > 0:
                selected = pending_messages[np.random.randint(len(pending_messages))-1]
                selected[0].send_ms_msg(selected[1])
                print selected[0],"-->",selected[1]
            else:
                print "no more pending messages..."
                break
            

    def initialize_node_list(self):
        for node in self.node_list:
            if node in self.variable_list:
                if len(self.variable_list[node].neighbours) == 1:
                    self.variable_list[node].pending.add(self.variable_list[node].neighbours[0])
                self.ordered_node_list.append(self.variable_list[node])
            else:
                if len(self.factor_list[node].neighbours) == 1:
                    self.factor_list[node].pending.add(self.factor_list[node].neighbours[0])
                self.ordered_node_list.append(self.factor_list[node])


# nt = Factor_Graph()
# nt.variable_list = {}
# nt.factor_list = {}
# nt.node_list = []
# nt.ordered_node_list = []
# # variables and factors in the right order in order to run sum-product algorithm
# nt.node_list = ["F_Influenza", "F_Smokes", "SoreThroat", "Fever", "Coughing", "Wheezing", 
# "F_SoreThroat", "F_Fever", "F_Coughing", "F_Wheezing", "Influenza", "Smokes", "F_Bronchitis", "Bronchitis"]
# # variables
# nt.variable_list["Influenza"] = Variable("Influenza", 2)
# nt.variable_list["Smokes"] = Variable("Smokes", 2)
# nt.variable_list["SoreThroat"] = Variable("SoreThroat", 2)
# nt.variable_list["Fever"] = Variable("Fever", 2)
# nt.variable_list["Bronchitis"] = Variable("Bronchitis", 2)
# nt.variable_list["Coughing"] = Variable("Coughing", 2)
# nt.variable_list["Wheezing"] = Variable("Wheezing", 2)
# # factself.ors
# nt.factor_list["F_Influenza"] = Factor("F_Influenza", 
#     np.array([0.05, 0.95]), 
#     [nt.variable_list["Influenza"]])
# nt.factor_list["F_Smokes"] = Factor("F_Smokes", 
#     np.array([0.2, 0.8]), 
#     [nt.variable_list["Smokes"]])
# nt.factor_list["F_SoreThroat"] = Factor("F_SoreThroat", 
#     np.array([[0.3, 0.001],[ 0.7, 0.999]]), 
#     [nt.variable_list["SoreThroat"], nt.variable_list["Influenza"]])
# nt.factor_list["F_Fever"] = Factor("F_Fever", 
#     np.array([[0.9, 0.05],[ 0.1, 0.95]]), 
#     [nt.variable_list["Fever"], nt.variable_list["Influenza"]])
# nt.factor_list["F_Bronchitis"] = Factor("F_Bronchitis", 
#     np.array([[[0.99, 0.9],[ 0.7, 0.0001]],[[ 0.01, 0.1],[ 0.3, 0.9999]]]) , 
#     [nt.variable_list["Bronchitis"],nt.variable_list["Influenza"], nt.variable_list["Smokes"]])
# nt.factor_list["F_Coughing"] = Factor("F_Coughing", 
#     np.array([[0.8, 0.07],[ 0.2, 0.93]]), [nt.variable_list["Coughing"], 
#     nt.variable_list["Bronchitis"]])
# nt.factor_list["F_Wheezing"] = Factor("F_Wheezing", 
#     np.array([[0.6, 0.001],[ 0.4, 0.999]]), [nt.variable_list["Wheezing"], 
#     nt.variable_list["Bronchitis"]])

# nt.initialize_node_list()
# nt.max_sum()
# nt.sum_product()



# Load the image and binarize
im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
imshow(im)
gray()
# Add some noise
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
figure()
imshow(noise_im)


nt2 = Factor_Graph()
nt2.variable_list = {}
nt2.factor_list = {}
nt2.node_list = []
nt2.ordered_node_list = []
noise_test_im = noise_im
for x in xrange(noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.variable_list[(1,x,y)] = Variable(str((1,x,y)), 2)
        if noise_test_im[x,y]:
            nt2.variable_list[(1,x,y)].set_observed(1)
        else:
            nt2.variable_list[(1,x,y)].set_observed(1)
        nt2.variable_list[(3,x,y)] = Variable(str((3,x,y)), 2)
        nt2.factor_list[(0,x,y)] = Factor(str((0,x,y)), 
            np.array([0.5, 0.5]), 
            [nt2.variable_list[(1,x,y)]])
        nt2.factor_list[(2,x,y)] = Factor(str((2,x,y)), 
            np.array([[0.9, 0.1],[ 0.1, 0.9]]), 
            [nt2.variable_list[(1,x,y)], nt2.variable_list[(3,x,y)]])
        if x > 0:
            nt2.factor_list[(x,y, x-1, y)] = Factor(str((x,y, x-1, y)), np.array([[0.9, 0.1],[ 0.1, 0.9]]), [nt2.variable_list[(3,x,y)], nt2.variable_list[(3,x-1,y)]])
        if y > 0:
            nt2.factor_list[(x,y, x, y-1)] = Factor(str((x,y, x, y-1)), np.array([[0.9, 0.1],[ 0.1, 0.9]]), [nt2.variable_list[(3,x,y)], nt2.variable_list[(3,x,y-1)]])

for x in xrange(noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.node_list.append((0,x,y))
for x in xrange(noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.node_list.append((1,x,y))
for x in xrange(noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.node_list.append((2,x,y))
for x in xrange(noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.node_list.append((3,x,y))
for x in xrange(1, noise_test_im.shape[0]):
    for y in xrange(noise_test_im.shape[1]):
        nt2.node_list.append((x,y, x-1, y))
for x in xrange(noise_test_im.shape[0]):
    for y in xrange(1, noise_test_im.shape[1]):
        nt2.node_list.append((x,y, x, y-1))

nt2.initialize_node_list()
nt2.loopy_max_sum(1000000)