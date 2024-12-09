from sklearn.model_selection import train_test_split
import idx2numpy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
rng = np.random.default_rng(51)
import pickle

def im2col(input_data, filter_size, stride=1, padding=0):
    batch_size, channel, input_height, input_width = input_data.shape
    output_height = (input_height + 2 * padding - filter_size) // stride + 1
    output_width = (input_width + 2 * padding - filter_size) // stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    input_col = np.zeros((batch_size, channel, filter_size, filter_size, output_height, output_width))
    for h in range(filter_size):
        height_max = h + stride * output_height
        for w in range(filter_size):
            width_max = w + stride * output_width
            input_col[:, :, h, w, :, :] = img[:, :, h:height_max:stride, w:width_max:stride]
    input_col = input_col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * output_height * output_width, -1)
    return input_col

def col2im(input_col, input_shape, filter_size, stride=1, padding=0):
    batch_size, channel, input_height, input_width = input_shape
    output_height = (input_height + 2 * padding - filter_size) // stride + 1
    output_width = (input_width + 2 * padding - filter_size) // stride + 1
    input_col = input_col.reshape(batch_size, output_height, output_width, channel, filter_size, filter_size).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((batch_size, channel, input_height + 2*padding + stride - 1, input_width + 2*padding + stride - 1))
    for h in range(filter_size):
        height_max = h + stride * output_height
        for w in range(filter_size):
            width_max = w + stride * output_width
            img[:, :, h:height_max:stride, w:width_max:stride] += input_col[:, :, h, w, :, :]
    return img[:, :, padding:input_height + padding, padding:input_width + padding]

def get_batches(data, batch_size):
  n = len(data)
  get_X = lambda z: z[0]
  get_y = lambda z: z[1]
  for i in range(0, n, batch_size):
    batch = data[i:i+batch_size]
    yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])

class Conv2D:
    def __init__(self, channel, count_filters, filter_size, stride, padding, activate):
        self.count_filters = count_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activate = activate
        self.channel = channel
        self.weights = np.random.standard_normal((channel, count_filters, filter_size, filter_size)) * 0.01
        self.biases = np.random.standard_normal((1, count_filters)) * 0.01
  
        self.input_col = None
        self.weight_col = None
        
        self.dW = None
        self.db = None

        self.__input_shape = None
        
    def forward(self, input):
        self.__input_shape = input.shape
        batch_size, _, input_height, input_width = input.shape
        output_height = (input_height + 2 * self.padding - self.filter_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.filter_size) // self.stride + 1
        input_col = im2col(input, self.filter_size, self.stride, self.padding)
        weight_col = self.weights.reshape(self.channel, -1).T.reshape(self.count_filters, self.channel * self.filter_size * self.filter_size).T
        self.output = np.dot(input_col, weight_col) + self.biases
        self.output = self.activate.get(self.output.reshape(batch_size, output_height, output_width, -1).transpose(0, 3, 1, 2))
        self.input_col = input_col
        self.weight_col = weight_col
        return self.output

    def backward(self, pred_delta, update=True):
        batch_size, delta_channel, delta_height, delta_width = pred_delta.shape
        delta = self.activate.dget(self.output) * pred_delta
        delta = delta.transpose(0, 2, 3, 1).reshape(batch_size * delta_height * delta_width, delta_channel)
        if update:
            self.db = np.sum(delta, axis=0)
            self.dW = np.dot(self.input_col.T, delta)
            self.dW = self.dW.transpose(1, 0)
            self.dW = self.dW.reshape(self.channel, self.count_filters, self.filter_size, self.filter_size)
        delta_col = np.dot(delta, self.weight_col.T)
        delta = col2im(delta_col, self.__input_shape, self.filter_size, self.stride, self.padding)
        return delta
    
    def get_weights(self):
        return [self.weights]
    
    def get_biases(self):
        return [self.biases]
    
    def update_weights(self, dW, db):
        self.weights -= dW[0]
        self.biases -= db[0]

    def get_dW(self):
        return self.dW
    
    def get_db(self):
        return self.db

class MaxPooling2D:
    def __init__(self, pool_size, strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size

        self.__input_shape = None
        self.mask = None

    def forward(self, input):
        self.__input_shape = input.shape
        batch_size, channel, input_height, input_width = self.__input_shape
        output_height = (input_height - self.pool_size) // self.strides + 1
        output_width = (input_width - self.pool_size) // self.strides + 1
        windows = np.lib.stride_tricks.as_strided(input,
                     shape=(batch_size, channel, output_height, output_width, self.pool_size, self.pool_size),
                     strides=(input.strides[0], input.strides[1],
                              self.strides * input.strides[2],
                              self.strides * input.strides[3],
                              input.strides[2], input.strides[3])
                     )
        output = np.max(windows, axis=(4, 5))

        maxs = output.repeat(2, axis=2).repeat(2, axis=3)
        x_window = input[:, :, :output_height * self.strides, :output_width * self.strides]
        self.mask = np.equal(x_window, maxs).astype(int)
        self.output = output
        return output
        
    def backward(self, pred_delta, update=True):
        delta_conv = pred_delta.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)
        delta_masked = np.multiply(delta_conv, self.mask)
        delta = np.zeros(self.__input_shape)
        delta[:, :, :delta_masked.shape[2], :delta_masked.shape[3]] = delta_masked
        return delta
    
    def get_weights(self):
        return []
    
    def get_biases(self):
        return []
    
    def update_weights(self, dW, db):
        pass

    def get_dW(self):
        return []
    
    def get_db(self):
        return []

class Flatten:
    def __init__(self):
        self.__input_shape = None
        self.output = None

    def forward(self, input):
        self.__input_shape = input.shape
        c, f, h, w  = input.shape
        return input.reshape(c, f * h * w)
    
    def backward(self, pred_delta, pred_weights=None, update=True):
        if pred_weights is None:
            return pred_delta.reshape(self.__input_shape)
        return np.matmul(pred_delta, pred_weights).reshape(self.__input_shape)
    
    def get_weights(self):
        return []
    
    def get_biases(self):
        return []
    
    def update_weights(self, dW, db):
        pass

    def get_dW(self):
        return []
    
    def get_db(self):
        return []
    
class Linear:
    def __init__(self, input_size, count, activate):
        self.count = count
        self.weights = np.random.standard_normal((count, input_size))  * 0.01
        self.biases = np.random.standard_normal((1, count)) * 0.01
        self.activate = activate

        self.input = None
        self.output = None

        self.dW = None
        self.db = None

    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.weights.T) + self.biases
        self.output = self.activate.get(self.output)
        return self.output
    
    def backward(self, pred_delta, pred_weights=None):
        new_delta = None
        if pred_weights is None:
            new_delta = pred_delta
        else:
            new_delta = self.activate.dget(self.output) * np.matmul(pred_delta, pred_weights)

        self.dW = np.matmul(new_delta.T, self.input)
        if len(new_delta.shape) == 1:
            self.db = new_delta
        else:
            self.db = np.sum(new_delta, axis=0)
        return new_delta

    def update_weights(self, dW, db):
        self.weights -= dW[0]
        self.biases -= db[0]

    def get_weights(self):
        return [self.weights]
    
    def get_biases(self):
        return [self.biases]
    
    def get_dW(self):
        return self.dW
    
    def get_db(self):
        return self.db

class Softmax:
    def get(input):
        e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def dget(input):
        s = input.reshape(-1, 1)
        return np.diagflat(input) - np.dot(s, s.T)

class ReLu:
    def get(input):
        return np.maximum(0, input)
    
    def dget(input):
        return (input > 0).astype(np.float64)

class CrossEntropy:
    def get(y_pred, y):
        return -np.sum(y * np.log(y_pred)) / y.shape[0]

    def dget(y_pred, y):
        return y_pred - y
    
class MSE:
    def get(y_pred, y):
        return np.mean(np.square(y_pred - y))

    def dget(y_pred, y):
        n = y.shape[0]
        return (2/n)*(y_pred - y)
    
class Adam:
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
        self.lr = lr
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.eps=eps

        self.t = 1
        self.mW = None
        self.mb = None
        self.vW = None
        self.vb = None

    def init_weight(self, weights, biases):
        self.mW = [np.zeros_like(w) for w in weights]
        self.mb = [np.zeros_like(b) for b in biases]
        self.vW = [np.zeros_like(w) for w in weights]
        self.vb = [np.zeros_like(w) for w in biases]

    def calc_step(self, dW, db, *args):
        i = args[0]
        if len(dW) > 0:
            self.mW[i] = self.beta_1*self.mW[i] + (1-self.beta_1)*dW
            self.mb[i] = self.beta_1*self.mb[i] + (1-self.beta_1)*db

            self.vW[i] = self.beta_2*self.vW[i] + (1-self.beta_2)*(dW**2)
            self.vb[i] = self.beta_2*self.vb[i] + (1-self.beta_2)*(db**2)

            mW_corr = self.mW[i] / (1-self.beta_1**self.t)
            mb_corr = self.mb[i] / (1-self.beta_1**self.t)
            vW_corr = self.vW[i] / (1-self.beta_2**self.t)
            vb_corr = self.vb[i] / (1-self.beta_2**self.t)
            
            if i % (len(self.mW) - 1) == 0:
                self.t += 1
            return self.lr*mW_corr / (np.sqrt(vW_corr)+self.eps), self.lr*mb_corr / (np.sqrt(vb_corr)+self.eps)
        else:
            return None, None

class LeNet5:
    def __init__(self, loss, method):
        self.loss = loss
        self.method = method

        self.layers = [
            Conv2D(1, 4, 5, 1, 0, ReLu), 
            MaxPooling2D(2),
            Conv2D(4, 12, 5, 1, 0, ReLu),
            MaxPooling2D(2),
            Flatten(),
            Linear(192, 120, ReLu),
            Linear(120, 84, ReLu),
            Linear(84, 10, Softmax)
        ]
        

    def _feedforward(self, X):
        inputs = X
        for i in range(len(self.layers)):
            inputs = self.layers[i].forward(inputs)
        return inputs
    
    def _backprop(self, y):
        delta = self.loss.dget(self.layers[-1].output, y)
        for i in range(len(self.layers)-1, -1, -1):
            if i != len(self.layers) - 1:
                if (type(self.layers[i]) != Conv2D) and (type(self.layers[i]) != MaxPooling2D):
                    delta = self.layers[i].backward(delta, self.layers[i+1].weights)
                else:
                    delta = self.layers[i].backward(delta)
            else:
                delta = self.layers[i].backward(delta)

    def train(self, X, y, X_test, y_test, epochs=1, batch_size=32):
        weights = []
        biases = []
        for i in range(len(self.layers)):
            weights.append(self.layers[i].get_weights())
            biases.append(self.layers[i].get_biases())

        self.method.init_weight(weights, biases)

        epoch_losses = np.array([])
        dataset = list(zip(X, y))
        rng.shuffle(dataset)
        
        plt.ion()
        for i in tqdm(range(epochs)):
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                X_batch_ex =  np.expand_dims(X_batch, axis=1)
                self._feedforward(X_batch_ex)
                self._backprop(y_batch)
                self._update_params()
            epoch_losses = np.append(epoch_losses, self._compute_loss(X_test, y_test))
            plt.plot(epoch_losses, c='red')
            plt.draw()
            plt.gcf().canvas.flush_events()
            plt.pause(0.01)
        plt.ioff()
        return epoch_losses

    def _compute_loss(self, X, y):
        y_pred = self.predict(X, True)
        return self.loss.get(y_pred, y)

    def predict(self, X, loss=False):
        X = np.expand_dims(X, axis=1)
        self._feedforward(X)
        pred = self.layers[-1].output
        if loss:
            return pred
        else:
            return np.argmax(pred, axis=1)
    
    def _update_params(self):
        for i in range(len(self.layers)):
            grad_dw = self.layers[i].get_dW()
            grad_db = self.layers[i].get_db()
            dW, db = self.method.calc_step(grad_dw, grad_db, i)
            self.layers[i].update_weights(dW, db)

def read_data(image_file, label_file):
    X = idx2numpy.convert_from_file(image_file)
    y = idx2numpy.convert_from_file(label_file)

    new_y = np.zeros((y.shape[0], 10))
    for i in range(new_y.shape[0]):
        for j in range(new_y.shape[1]):
            if y[i] == j:
                new_y[i, j] = 1

    y = new_y

    return X, y

if __name__ == '__main__':
    image_file = 'data/train-images-idx3-ubyte'
    label_file = 'data/train-labels-idx1-ubyte'

    X, y = read_data(image_file, label_file)

    X_train, X_test, y_train, y_test = train_test_split(X[0:3000], y[0:3000], test_size=0.3, random_state=42)

    method = Adam(lr=1e-4)
    model = LeNet5(CrossEntropy, method)
    epoch_losses = model.train(X_train, y_train, X_test, y_test, epochs=50)
    print(f"First epoch loss - {epoch_losses[0]}\nLast epoch loss - {epoch_losses[-1]}")
    plt.show()
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)