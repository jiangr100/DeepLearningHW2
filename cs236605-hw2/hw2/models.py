import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        
        prev = in_features
        
        for (i, cur_fea) in enumerate(hidden_features):
            fc = Linear(prev, cur_fea)
            if activation == 'relu':
                activation_layer = ReLU()
            else:
                activation_layer = Sigmoid()
            prev = cur_fea
            blocks.extend([fc, activation_layer])
            if dropout > 0:
                dropout_layer = Dropout(dropout)
                blocks.append(dropout_layer)
        
        fc = Linear(prev, num_classes)
        blocks.append(fc)
        
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        
        N = len(self.filters)
        P = self.pool_every
        
        prev = in_channels
        it = 0
        
        for i in range(int(N/P)):
            for j in range(P):
                conv = nn.Conv2d(prev, self.filters[it], kernel_size=3, padding=1)
                relu = nn.ReLU()
                layers.extend([conv, relu])
                
                prev = self.filters[it]
                it += 1
                
            maxpool = nn.MaxPool2d(2, stride=2)
            layers.append(maxpool)

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        
        M = len(self.hidden_dims)
        N = len(self.filters)
        P = self.pool_every
        last_layer_channels = self.filters[-1]
        
        prev = int(last_layer_channels*in_h*in_w / (2**(int(N/P)))**2)
        
        for i in range(M):
            linear = nn.Linear(prev, self.hidden_dims[i])
            relu = nn.ReLU()
            layers.extend([linear, relu])
            
            prev = self.hidden_dims[i]
        
        linear = nn.Linear(prev, self.out_classes)
        layers.append(linear)
        
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        
        out1 = self.feature_extractor(x)
        out1 = out1.reshape(out1.shape[0], -1)
        out = self.classifier(out1)
        
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
        
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        
        N = len(self.filters)
        P = self.pool_every
        
        prev = in_channels
        
        for i in range(int(N/P)):
            for j in range(P):
                conv = nn.Conv2d(prev, self.filters[j], kernel_size=3, padding=1)
                relu = nn.ReLU()
                dropout = nn.Dropout2d(0.5)
                layers.extend([conv, relu, dropout])
                
                prev = self.filters[j]
            
            batchnorm = nn.BatchNorm2d(prev)
            maxpool = nn.MaxPool2d(2, stride=2)
            layers.extend([batchnorm, maxpool])

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        
        M = len(self.hidden_dims)
        N = len(self.filters)
        P = self.pool_every
        last_layer_channels = self.filters[-1]
        
        prev = int(last_layer_channels*in_h*in_w / (2**(int(N/P)))**2)
        
        for i in range(M):
            linear = nn.Linear(prev, self.hidden_dims[i])
            relu = nn.ReLU()
            dropout = nn.Dropout(0.5)
            layers.extend([linear, relu, dropout])
            
            prev = self.hidden_dims[i]
        
        #batchnorm = nn.BatchNorm1d(1) ??
        linear = nn.Linear(prev, self.out_classes)
        layers.append(linear)
        
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        
        out1 = self.feature_extractor(x)
        out1 = out1.reshape(out1.shape[0], -1)
        out = self.classifier(out1)
        
        # ========================
        return out
    
    # ========================

