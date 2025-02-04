import numpy as np
import random
import copy


class NetParams:
    def __init__(self, sizes, cost_type):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.cost_type = cost_type.lower()
    
    def set_io(self, input_size, output_size):
        self.sizes.insert(0, input_size)
        self.sizes.append(output_size)
        self.num_layers += 2


class Network(object):
    def __init__(self, params: NetParams) -> None:
        self.sizes = params.sizes
        self.num_layers = params.num_layers
        self.cost_type = params.cost_type
        self.biases = [0.1 * np.random.randn(i, 1) for i in self.sizes[1:]] # (i,1) to create column vector, (i) is 1D row vector
        self.weights = [0.1 * np.random.randn(i, j) for j, i in zip(self.sizes[:-1], self.sizes[1:])]

    def get_params(self):
        return {'biases': self.tolist(self.biases), 'weights': self.tolist(self.weights)}
    
    def tolist(self, arr):
        return [x.tolist() for x in arr]
    
    def tonparray(self, arr):
        return [np.array(x) for x in arr]
    
    def update(self, nn):
        self.biases = copy.deepcopy(nn.biases)
        self.weights = copy.deepcopy(nn.weights)
    
    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = relu(np.dot(w, a) + b)
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return a
    
    def train(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """
        train_data: [(x0, y0), (x1, y1), ... (xn, yn)]
                    where x0, x1, ... xn are vectors of input activations
                    and y0, y1, ... yn are vectors of output activations
        """
        n = len(train_data) # size of training data
        prev_res = 0 # used for accuracy change calculation

        for epoch in range(epochs):
            # decompose inputs into random mini batches
            # random.shuffle(train_data)
            # mini_batches = [train_data[i:i+mini_batch_size] for i in range(0,n,mini_batch_size)]
            # train on each mini batch
            # for mini_batch in mini_batches:
            self.update_mini_batch(train_data, eta)
            
            # print results
            # if test_data is not None:
            #     cur_res = self.test(test_data)
            #     change = "INF" if prev_res == 0 else (cur_res - prev_res) / prev_res
            #     print(f"Epoch {epoch+1}: {cur_res}/{len(test_data)}, {change}% change")
            #     prev_res = cur_res
            # else:
            #     print(f"Epoch {epoch+1} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        """
        mini_batch: subset of train_data
        """
        # print(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases] # ∇b
        nabla_w = [np.zeros(w.shape) for w in self.weights] # ∇w

        # sum of ∇w = ∑ ∇w_j = ∑ ∂C/∂w
        # same for b
        for x, y in mini_batch:
            # print(f"{x}\n{y}")
            cur_nb, cur_nw = self.backprop(x, y) # calculate ∂C/∂w and ∂C/∂b
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, cur_nb)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, cur_nw)]
        
        # m = size of mini batch
        # ∇w = - 1/m * η * ∑ ∂C/∂w
        # w = w + ∇w = w - η/m*∑∂C/∂w
        # same for b
        self.weights = [
            w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w) 
        ]
        self.biases = [
            b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)
        ]
    
    def backprop(self, x, y):
        """
        x: vector of input activations
        y: vector of output activations
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        a = x # vector of input activations
        activations = [x]
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            zs.append(z)
            a = relu(z)
            activations.append(a)
        z = np.dot(self.weights[-1], z) + self.biases[-1]
        zs.append(z)
        # a = np.dot(self.weights[-1], a) + self.biases[-1]
        a = z
        activations.append(a)
        
        # calculate output layer errors
        # delta = error
        delta = self.cost_deriv(activations[-1], y) * 1
        # print(delta)
        # store gradients for output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # backpropagate
        for layer in range(2, self.num_layers):
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * relu_deriv(zs[-layer])
            # store gradients for current layer
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        
        return (nabla_b, nabla_w)


    def cost_deriv(self, a, y):
        # returns vector of partial cost derivatives ∂C/∂a
        # only for output layer

        # Mean squared error
        if self.cost_type == 'mse':
            return (a - y)
        
        # Cross-entropy
        if self.cost_type == 'ce':
            return (y/a + (1-y)/(1-a))

    def test(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        num_correct = sum(int(x == y) for (x, y) in results)
        return num_correct


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
    return x * (x > 0)

def relu_deriv(x):
    return 1 * (x > 0)

# x = Network(NetParams([1,6,3], 'mse')).__str__()
# print(x)
# from data_transfer import *
# save('net.json', x)
# import mnist_loader

# tr_d, va_d, te_d = mnist_loader.load_data_wrapper()
# net = Network([784, 30, 10], cost_type='mse')
# net.train(tr_d, 30, 10, 3.0, test_data=te_d)
if __name__ == '__main__':
    nn = Network(NetParams([1,6,1], 'mse'))
    print("before:", nn.feedforward(np.array([[2]])))
    nn.train([(np.array([[2]]), np.array([[-1.5]])) for _ in range(200)], 1000, mini_batch_size=500, eta=0.01)
    print("after:", nn.feedforward(np.array([[2]])))