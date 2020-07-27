import numpy as np
import h5py
import os
import matplotlib.pyplot as plt



EPSILON = 1e-12

'''Define class for activation functions'''
class Activation(object):
    
    def relu(self,x):
        return np.maximum(x,0)  

    def relu_deriv(self,a):
        # a = np.maximum(x,0)
        a[a<0] = 0
        a[a>=0] = 1
        return a
    
    def leaky_relu (self,x):
        return np.maximum(x,0.1*x)

    def leaky_relu_deriv (self,a):
        # a = np.maximum(x,0.1*x)
        a[a<0] = 0.1
        a[a>=0] = 1
        return a
    
    def softmax(self,x):
        # Normalise the input to prevent overflow problem in np.exp
        x_max = x.max(axis = 1, keepdims = True)
        x_norm = x-x_max
        norms = np.sum(np.exp(x_norm), axis = 1, keepdims=True)
        return np.exp(x_norm)/(norms)
   
    def softmax_deriv(self, a):
        #a = np.exp(x) / np.sum(np.exp(x), axis=0)
        return a * (1 - a )
        
    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, a):
        # a = np.tanh(x)   
        return 1.0 - a**2
    
    def logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def logistic_deriv(self, a):
        # a = logistic(x) 
        return  a * (1 - a )      
            
    def __init__(self,activation='relu'):
        if activation == 'logistic':
            self.f = self.logistic
            self.f_deriv = self.logistic_deriv
        elif activation == 'softmax':
            self.f = self.softmax
            self.f_deriv = self.softmax_deriv          
        elif activation == 'tanh':
            self.f = self.tanh
            self.f_deriv = self.tanh_deriv
        elif activation == 'relu':
            self.f = self.relu
            self.f_deriv = self.relu_deriv
        elif activation == 'leaky relu':
            self.f = self.leaky_relu
            self.f_deriv = self.leaky_relu_deriv


class HiddenLayer(object):
    
    def __init__(self,n_in, n_out,
                 activation_last_layer='tanh',activation='tanh', W=None, b=None,
                 batch_norm = True, dropout_fraction = 0.0):
        
        self.inputs=None
        self.activation=Activation(activation).f
        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        
        # Uniformly sampled with a variance of 6/(n_in+n_out)
        
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out))
        self.b = np.zeros(n_out,)

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        #Intialise velocities of W,b for Momentum & Adam updates, same shape as W,b
        self.velocity_W = np.zeros(self.W.shape)
        self.velocity_b = np.zeros(self.b.shape)
        # ADAM
        self.sqr_grad_W = np.zeros(self.W.shape) 
        self.sqr_grad_b = np.zeros(self.b.shape) 
        
        # Initial estimates of batch norm parameters
        self.batch_norm = batch_norm # T or F
        self.gamma = np.random.randn(1, n_out)
        self.beta = np.random.rand(1, n_out)
        self.mean_training = np.zeros((1, n_out))
        self.var_training = np.zeros((1, n_out))
        
        #Dropout vector
        self.dropout_fraction = dropout_fraction
        self.dropout_vector = np.random.binomial(1, 1-self.dropout_fraction, size = (1, n_out))
    
    def forward_batch(self, x):
        """
        x: The pre-activation of this layer, shape (batch_size, n_out) 
        """
        # Mean of the mini-batch
        mu = np.mean(x, axis=0)
        # VARIANCE of the mini-batch
        var = np.var(x, axis=0)

        self.std_inv = 1.0/np.sqrt(var + EPSILON)
        # The normalized input, x_hat
        self.x_hat = (x - mu)*self.std_inv
        # Batch normalizing (affine) transformation
        y = self.gamma*self.x_hat + self.beta
        return y, mu, var
    
    def backward_batch(self, delta):
        """
        This function calculates the dL/dparams for params of batch norm layer
        Also calculates dx = dL/dx where x are pre activations of this layer
        :delta: shape (batch_size, n_out)
        """
        m, d = delta.shape
        dx_hat = delta*self.gamma # shape (batch_size, n_out)
        dx = self.std_inv * (dx_hat - np.mean(dx_hat, axis=0) - self.x_hat*np.mean(dx_hat*self.x_hat, axis=0))
        self.dgamma = np.sum(delta*self.x_hat, axis=0) #(axis = 0) sums over batches,result is across columns
        self.dbeta = np.sum(delta, axis=0)
        return dx
        
    def forward(self, inputs, training = True):
        '''
        inputs: a symbolic tensor of shape (batch_size, n_in)
        '''
        lin_output = np.matmul(inputs, self.W) + self.b
        
        if training == True:
            self.inputs=inputs 
            if self.batch_norm == True:
                y, mu, var = self.forward_batch(lin_output)
                # For testing later on
                self.mean_training = 0.9 * self.mean_training + 0.1*mu
                self.var_training = 0.9 * self.var_training + 0.1*var
                self.output = (y if self.activation is None
                               else self.activation(y))
            else: # No batch_norm
                self.output = (lin_output if self.activation is None
                               else self.activation(lin_output))
                
            if self.dropout_fraction > 0.0:
                self.output = self.dropout_vector*self.output
            return self.output
                
        else:
            if self.batch_norm == True:
                x_hat = (lin_output - self.mean_training)*(1.0/np.sqrt(self.var_training+EPSILON))
                y = self.gamma*x_hat + self.beta
                output = (y if self.activation is None 
                          else self.activation(y))
            else:
                output = lin_output
            return output
            
    
    def backward(self, delta, output_layer=False):
        """
        delta: shape (batch_size, n_out). If batch_norm = True, then delta is dL/dx_hat
        Otherwise, its dL/dx where x are preactivations of this layer)
        self.inputs: (inputs to this layer), shape (batch_size, n_in)
        """
        if self.batch_norm == True:
            delta = self.backward_batch(delta) #dLoss/d(preactivations of this layer)
            
        # Same size as W, shape (n_in, n_out)
        self.grad_W = np.atleast_2d(self.inputs).T.dot(np.atleast_2d(delta))
        # Shape (n_out,)
        self.grad_b = np.sum(delta, axis = 0) # Change from mean to sum
        if self.activation_deriv:
            delta = delta.dot(self.W.T)*self.activation_deriv(self.inputs)
        return delta


def classification_accuracy(classification_scores, true_labels):
    """
    Returns the fractional classification accuracy for a batch of N predictions.

    Parameters
    ----------
    classification_scores : numpy.ndarray, shape=(N, K)
        The scores for K classes, for a batch of N pieces of data
        (e.g. images).
    true_labels : numpy.ndarray, shape=(N,K)

    Returns
    -------
    float: (num_correct) / N
    """
    true_labels = np.argmax(true_labels, axis = -1)
    return np.mean(np.argmax(classification_scores, axis=1) == true_labels)


def early_stop(acc_list):
    """
    Function for early stopping. Stops if percentage change from previous validation accuracy
    decreases by 5%
    """
    if len(acc_list) <= 1:
        return False
    else:
        perc_change_from_previous = (acc_list[-2] - acc_list[-1])/(acc_list[-1])
        if perc_change_from_previous <= -0.05:
            return True
        else:
            return False


class MLP:
    
    def __init__(self, layers, activation=[None,'tanh','tanh'],
                 batch_norm = [None, True, True], dropouts = [0.0, 0.0, 0.0]):
        """
        :param layers: A list containing the number of units in each layer.Should be at least two values
        :param activation: The activation function to be used. Can be "logistic" or "tanh"
        """
        ### initialize layers
        self.layers=[]
        self.activation=activation
        self.batch_norms = batch_norm
        
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1],
                                           batch_norm = batch_norm[i+1], dropout_fraction = dropouts[i+1]))
            
    def forward(self,inputs, training = True):
        for layer in self.layers:
            output=layer.forward(inputs, training = True)
            inputs=output
        return output

    #Cross Entropy
    def cross_entropy(self, y_true, logits):
        """
        logits (batch_size x num_classes)
        y_true is labels (batch_size x num_classes) Note that y is an one-hot encoded vector. 
        """
        # Batch size
        m = y_true.shape[0]
        # Shape (batch_size, num_classes)
        log_likelihood_matrix = -1*y_true*np.log(logits+EPSILON)
        # Shape (1)
        loss = np.sum(log_likelihood_matrix)/m
        # Shape (batch_size, num_classes)
        delta = logits-y_true
        return loss, delta

    def backward(self,delta):
        delta = self.layers[-1].backward(delta,output_layer=True)
        # from last layer to first layer
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    #Gradient Descent parameters W,b updates
    def update(self, learning_rate=0.001,beta1=0.9,beta2 = 0.999,weight_decay =0.0,optimizer=None):
        """
        :learning_rate: learning rate (float)
        :beta1, beta2: beta values (float) for Momentum & Adam updates
        :optimizer: Gradient Descent optimizer, can be "momentum", "adam" or None
        """
        for layer in self.layers:
            if optimizer == 'momentum':
                # Update velocities of W,b
                layer.velocity_W = beta1*layer.velocity_W + (1-beta1)*layer.grad_W
                layer.velocity_b = beta1*layer.velocity_b + (1-beta1)*layer.grad_b
                # Update W,b
                layer.W = layer.W - learning_rate*layer.velocity_W - weight_decay*layer.W 
                layer.b = layer.b - learning_rate*layer.velocity_b - weight_decay*layer.b
                
                if layer.batch_norm == True:
                    layer.gamma = layer.gamma - learning_rate*layer.dgamma
                    layer.beta = layer.gamma - learning_rate*layer.dbeta
                    
            elif optimizer == 'adam':
                # Update velocity of W,b
                layer.velocity_W = beta1*layer.velocity_W + (1-beta1)*layer.grad_W
                layer.velocity_b = beta1*layer.velocity_b + (1-beta1)*layer.grad_b
                # Correct the velocity of W,b
                velocity_corrected_W = layer.velocity_W / (1-beta1**2+EPSILON)  
                velocity_corrected_b = layer.velocity_b / (1-beta1**2+EPSILON)  
                # Calculate squared gradients of W,b
                layer.sqr_grad_W = beta2*layer.sqr_grad_W + (1-beta2)*np.power(layer.grad_W,2)
                layer.sqr_grad_b = beta2*layer.sqr_grad_b + (1-beta2)*np.power(layer.grad_b,2)
                # Correct the squared gradients of W,b
                sqr_grad_corrected_W = layer.sqr_grad_W / (1-beta2**2+EPSILON) 
                sqr_grad_corrected_b = layer.sqr_grad_b / (1-beta2**2+EPSILON) 
                # Update W,b
                layer.W = layer.W - learning_rate*(velocity_corrected_W/(np.sqrt(sqr_grad_corrected_W)+EPSILON)) - weight_decay*layer.W
                layer.b = layer.b - learning_rate*(velocity_corrected_b/(np.sqrt(sqr_grad_corrected_b)+EPSILON)) - weight_decay*layer.b
                
                if layer.batch_norm == True:
                    layer.gamma = layer.gamma-learning_rate*layer.dgamma
                    layer.beta = layer.gamma-learning_rate*layer.dbeta
            
            # Stochastic only
            else:
                layer.W = layer.W - learning_rate*layer.velocity_W - weight_decay*layer.W 
                layer.b = layer.b - learning_rate*layer.velocity_b - weight_decay*layer.b
                                
                if layer.batch_norm == True:
                    layer.gamma = layer.gamma - learning_rate*layer.dgamma
                    layer.beta = layer.gamma - learning_rate*layer.dbeta


    def fit(self,X_train,Y_train,X_val,Y_val,
            learning_rate=0.001,epochs=10,batch_size = 32,weight_decay=0.0,beta1 = 0.9,
            verbose = True, optimizer = None):
        """
        :param X_train, Y_train: train data and labels. Y is one hot encoded
        :param X_val, Y_val: validation 
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        epoch_losses = []
        train_acc_list = []
        val_acc_list = []

        for k in range(epochs):
            # Every epoch find loss
            loss_arr = []
            for i in range(0, len(X_train), batch_size):
                # forward pass
                logits = self.forward(X_train[i:i+batch_size])
                # backward pass
                loss, delta = self.cross_entropy(y_true = Y_train[i:i+batch_size],logits = logits)
                loss_arr.append(loss)
                self.backward(delta)
                # update
                self.update(learning_rate = learning_rate,
                            weight_decay = weight_decay,
                            beta1 = beta1,
                            optimizer = optimizer)
                
            train_accuracy = classification_accuracy(self.predict(X_train), Y_train)
            val_accuracy = classification_accuracy(self.predict(X_val), Y_val)
            epoch_losses.append(np.mean(loss_arr))
            train_acc_list.append(train_accuracy)
            val_acc_list.append(val_accuracy)
            
            if verbose == True:
                print('end of epoch {}'.format(k))
                print('training accuracy is {:.4f}'.format(train_accuracy))
                print('validation accuracy is {:.4f}'.format(val_accuracy))
                print('training loss is {:.4f}'.format(np.mean(loss_arr)))
                
            if early_stop(val_acc_list) == True or np.isnan(epoch_losses[-1]):
                break
            
        return epoch_losses, train_acc_list, val_acc_list
    
    def predict(self, X):
        predicted_logits = self.forward(X, training = False)
        return predicted_logits        