import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score
from scipy.special import expit
def softmax(r):
    shift=r-np.max(r)
    exps=np.exp(shift)
    return exps/np.sum(exps,axis=0)

def sigmoid(r):
    return expit(r)

def swish(r):
    beta=1
    return r * sigmoid(beta * r)

class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None):
        self.eps=1/np.sqrt(n_input)
        self.weights = np.random.normal(0,self.eps,(n_neurons, n_input))
        self.activation = activation
        self.n_input=n_input
        self.n_neurons=n_neurons
        self.z = np.zeros((n_neurons,1))
        self.del_z = np.zeros((n_neurons,1))
        self.last_activation = np.zeros((n_neurons,1))
        self.input = np.zeros((n_input,1))
        self.error = np.zeros([n_neurons,1])
        self.delta = np.zeros([n_neurons,n_input])
    def init_weight(self):
        self.delta_w=np.zeros((self.n_neurons, self.n_input))
    def activate(self, x,istest=False):
        if(istest):
            z=np.dot(self.weights,x) #+ self.bias
            act=self._apply_activation(z)
            return act
        else:
            self.input=x.reshape(-1,1)
            self.z = np.dot(self.weights,self.input) #+ self.bias
            self.last_activation=self._apply_activation(self.z)
            return self.last_activation
    def _apply_activation(self, r):
        # In case no activation function was chosen
        if self.activation is None:
            return r
        # tanh
        if self.activation == 'tanh':
            return np.tanh(r)
        if self.activation == 'relu':
            return np.maximum(r,np.zeros(r.shape))
        # sigmoid
        if self.activation == 'sigmoid':
            return sigmoid(r)
        # swish
        if self.activation == 'swish':
            return swish(r)
        # softmax
        if self.activation == 'softmax':
            return softmax(r)
        return r

    def apply_activation_derivative(self, r, y=None, output=None):
        if self.activation is None:
            return r
        if self.activation == 'tanh':
            return 1 - (np.tanh(r)) ** 2
        if self.activation == 'relu':
            return (r > 0.0)*1.0
        if self.activation == 'swish':
            beta=1
            k=swish(r)
            return beta * k + (sigmoid(beta * r)*(1-beta*k))
        if self.activation == 'sigmoid':
            k=sigmoid(r)
            return k * (1.0 - k)
        if self.activation == 'softmax':
            k=softmax(r)
            return k * (1.0 - k)
        return r
class NeuralNetwork:

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X, istest=False):
        for layer in self._layers:
            X = layer.activate(X,istest)
        return X

    def predict(self, X,istest=False):
        #predict
        forward_pass = self.feed_forward(X.T,istest)
        return np.argmax(forward_pass,axis=0)

    def backpropagation(self, X, Y, n_classes, learning_rate=1e-4,weight_reg=1e-4):
        #code for backprop and weight update after going through the full batch
        for layer in self._layers:
            layer.init_weight()
        # Loop over the layers backward
        for i in range(X.shape[0]):
            # Feed forward for the output
            input=X[i]
            y=np.eye(n_classes)[Y[i]].reshape((n_classes,))
            output = self.feed_forward(input).reshape((n_classes,))
            for l in reversed(range(len(self._layers))):
                layer = self._layers[l]
                # for the output layer
                if layer == self._layers[-1]:
                    #print(output.shape,y.shape)
                    err=-(y-output)
                    layer.error=err.reshape(-1,1)
                    # The output = layer.last_activation in this case
                    layer.del_z = layer.apply_activation_derivative(layer.z) * layer.error
                    #print("llayer delta",layer.z.shape,layer.error.shape)
                    layer.delta = np.dot(layer.del_z,layer.input.T)
                    #print("err",layer.error.shape,layer.delta.shape,layer.apply_activation_derivative(output).shape)
                else:
                    next_layer = self._layers[l + 1]
                    layer.error = np.dot(next_layer.weights.T, next_layer.del_z)
                    layer.del_z = layer.apply_activation_derivative(layer.z) * layer.error
                    layer.delta = np.dot(layer.del_z, layer.input.T)
                layer.delta_w += layer.delta

        # apply weights update from all samples in batch
        for l in range(len(self._layers)):
            layer = self._layers[l]
            layer.weights -= (layer.delta_w/i * learning_rate + weight_reg* layer.weights)
            #print("l",l,layer.weights.shape)
    def fit(self,X_train,y_train, n_classes, minibatch_size=10, learning_rate=1, max_epochs=1000):
        for epoch in range(max_epochs):
            X_train, y_train = shuffle(X_train, y_train)
            onehot=np.eye(n_classes)[y_train].reshape((n_classes,len(y_train)))
            for i in range(0, X_train.shape[0], minibatch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                X_train_mini = X_train[i:i + minibatch_size]
                y_train_mini = y_train[i:i + minibatch_size]
                self.backpropagation(X_train_mini, y_train_mini, n_classes, learning_rate)
            if epoch % (max_epochs/10) == 0:
                print("epoch",epoch)
                y_pred=self.predict(X_train,y_train)
                print("Training set accuracy",accuracy_score(y_train,y_pred))
        return self
    def predict(self,X_test,y_test):
        out=self.feed_forward(X_test.T,istest=True)
        y_pred=np.argmax(out,axis=0)
        #print("accuracy", accuracy_score(y_test,y_pred))
        return y_pred
