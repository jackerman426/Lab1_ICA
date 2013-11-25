import numpy as np

variable_list = {}
factor_parent_list = {}
factor_list = {}
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
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg
        if(len(self.in_msgs) == len(self.neighbours)):
            for neighbour in self.neighbours:
                self.pending.add(neighbour)
        if(len(self.in_msgs) == len(self.neighbours) - 1):
            receiver_node = None
            for neighbour in self.neighbours:
                if neighbour not in self.in_msgs:
                    receiver_node = neighbour
                    break
            self.pending.add(receiver_node)
    
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
        marginal = np.ones(2)
        for neighbour in self.neighbours:
            marginal = np.multiply(marginal, self.in_msgs[neighbour])
        if Z is None:
            Z = sum(marginal)
        marginal /= Z
        return marginal, Z
    
    def send_sp_msg(self, other):
        num_neighbours = len(self.neighbours)
        if num_neighbours == 1:
            message = np.ones(self.num_states)
            other.receive_msg(self, message)
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
                other.receive_msg(self, message)
                self.pending.remove(other)
                self.pending

    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass

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
        print "------factor--------", self.name
        print "--------"
        if len(self.neighbours) == 1:
            message = self.f
            other.receive_msg(self, message)
            print message
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
                other.receive_msg(self, np.array([0.5, 0.5]))
                self.pending.remove(other)
                print message

   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass

variable_list["Influenza"] = Variable("Influenza", 2)
variable_list["Smokes"] = Variable("Smokes", 2)
variable_list["SoreThroat"] = Variable("SoreThroat", 2)
variable_list["Fever"] = Variable("Fever", 2)
variable_list["Bronchitis"] = Variable("Bronchitis", 2)
variable_list["Coughing"] = Variable("Coughing", 2)
variable_list["Wheezing"] = Variable("Wheezing", 2)

factor_parent_list["Influenza"] = {}
factor_parent_list["Smokes"] = {}
factor_parent_list["SoreThroat"] = {"Influenza": None}
factor_parent_list["Fever"] = {"Influenza": None}
factor_parent_list["Bronchitis"] = {"Influenza": None, "Smokes":None}
factor_parent_list["Coughing"] = {"Bronchitis": None}
factor_parent_list["Wheezing"] = {"Bronchitis": None}

factor_list["Influenza"] = Factor("Influenza", np.array([0.05, 0.95]), [variable_list["Influenza"]])
factor_list["Smokes"] = Factor("Smokes", np.array([0.2, 0.8]), [variable_list["Smokes"]])
factor_list["SoreThroat"] = Factor("SoreThroat", np.array([[0.3, 0.001],[ 0.7, 0.999]]), [variable_list["SoreThroat"], variable_list["Influenza"]])
factor_list["Fever"] = Factor("Fever", np.array([[0.9, 0.05],[ 0.1, 0.95]]), [variable_list["Fever"], variable_list["Influenza"]])
factor_list["Bronchitis"] = Factor("Bronchitis", np.array([[[0.99, 0.9],[ 0.7, 0.0001]],[[ 0.01, 0.1],[ 0.3, 0.9999]]]) , [variable_list["Bronchitis"],variable_list["Influenza"], variable_list["Smokes"]])
factor_list["Coughing"] = Factor("Coughing", np.array([[0.8, 0.07],[ 0.2, 0.93]]), [variable_list["Coughing"], variable_list["Bronchitis"]])
factor_list["Wheezing"] = Factor("Wheezing", np.array([[0.6, 0.001],[ 0.4, 0.999]]), [variable_list["Wheezing"], variable_list["Bronchitis"]])

# level 0
factor_list["Influenza"].send_sp_msg(variable_list["Influenza"])
factor_list["Smokes"].send_sp_msg(variable_list["Smokes"])

# level 1
variable_list["Influenza"].send_sp_msg(factor_list["SoreThroat"])
variable_list["Influenza"].send_sp_msg(factor_list["Fever"])
variable_list["Influenza"].send_sp_msg(factor_list["Bronchitis"])
variable_list["Smokes"].send_sp_msg(factor_list["Bronchitis"])

#level 2
factor_list["SoreThroat"].send_sp_msg(variable_list["SoreThroat"])
factor_list["Fever"].send_sp_msg(variable_list["Fever"])
factor_list["Bronchitis"].send_sp_msg(variable_list["Bronchitis"])

#level 3
variable_list["Bronchitis"].send_sp_msg(factor_list["Coughing"])
variable_list["Bronchitis"].send_sp_msg(factor_list["Wheezing"])

#level 4
factor_list["Coughing"].send_sp_msg(variable_list["Coughing"])
factor_list["Wheezing"].send_sp_msg(variable_list["Wheezing"])