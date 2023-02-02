import numpy as np
from random import shuffle
import Backprop
import Metrics as mt
class MLP:
        
        def __init__(self, architecture_list):
                #initialize with network architecture provided as a argument.
                self.architecture = architecture_list
                self.number_of_layer = len(architecture_list)
                self.biases = []
                #initialize biases and weights with random normal distribution based on network arch.
                for x in architecture_list[1:]:
                        self.biases.append(np.random.randn(x,1))
                self.weights = []
                for y in range(self.number_of_layer - 1):
                        self.weights.append(np.random.randn(architecture_list[y+1], architecture_list[y]))
                
        def train(self, x_train, x_label, epoch, batch_size, eta, isClassification):
                #take inputs of training set, number of epoch, mini batch size and learning rate eta
                
                train_tuple = [(x_train[i], x_label[i]) for i in range(len(x_train))]
                #for validation, split trainset into train and validate
                #measure the performance of each epoch based on the validation
                epoc_list = []
                result_list = []
                split = int(len(train_tuple) * 0.8)
                validation_set = train_tuple[split:]
                train_tuple = train_tuple[:split]
                #modify weights and biases with each epoch. If validation set to true
                #check performance for each epoch
                for y in range(epoch):                        
                        batch_list = []
                        shuffle(train_tuple)
                        #divide data into mini batches to perform stocastic gradient descent.
                        for x in range(0, len(train_tuple), batch_size):
                                batch_list.append(train_tuple[x: x+ batch_size])
                        #update weights and biases for each batch 
                        for batch in batch_list:
                                self.update_weights_biases(batch, eta)
                        
                        #test validation score for each epoch
                        result = self.test_validation(validation_set, isClassification, y)
                        epoc_list.append(y)
                        result_list.append(result)
                #return validation score         
                return epoc_list, result_list
                        
        def update_weights_biases(self, batch, eta):
                delta_weights = []
                delta_biases = []
                #initialize weights and biases to 0 to hold the values for each instance 
                for x in self.biases:   delta_biases.append(np.zeros(x.shape))
                for y in self.weights:  delta_weights.append(np.zeros(y.shape))
                
                #call Backprop class to get delta error for each instances
                for x, y in batch:
                        d_b, d_w = Backprop.Backprop().backprop(x, y, self.biases, self.weights, self.number_of_layer)
                        #add delta error for a point in the local weights and biases list
                        for i in range(len(d_b)):
                                delta_biases[i] = delta_biases[i] + d_b[i]
                        for j in range(len(d_w)):
                                delta_weights[j] = delta_weights[j] + d_w[j]
                
                #update weights and biases based on the learning rate and error calculated for
                # all the mini batches data points average.                
                for i in range(len(self.biases)):
                        self.biases[i] = self.biases[i] - (eta/len(batch)) * delta_biases[i]
                for j in range(len(self.weights)):
                        self.weights[j] = self.weights[j] - (eta/len(batch) * delta_weights[j])
                        
                        
        def feed_forward(self, input_):
                #for a input run the network and get the output activation layer unit vector.
                activation = input_
                for b, w in zip(self.biases, self.weights):
                        activation = mt.Metrics().sigmoid(np.dot(w, activation) + b)
                return activation
        
        def test(self, x_test, isClassification):
                #test the model with test set and return predicted output for classification or regression.
                if isClassification:                        
                        predicted = [np.argmax(self.feed_forward(x)) for x in x_test]
                else:
                        predicted = [self.feed_forward(x)[0][0] for x in x_test]
                return predicted
        
        def test_validation(self, validate_set, isClassification, epoch):
                #test the model with validation set and return accuracy or rmse based on classification/regression.
                predicted = []
                label = []
                if isClassification:       
                        for x, y in validate_set:
                                predicted.append(np.argmax(self.feed_forward(x)))
                                label.append(np.argmax(y))
                        acc, prec, recall = mt.Metrics().confusion_matrix(label, predicted)
                        #print('Epoch {0} completed with acc::: {1}'.format(epoch, acc))
                        return acc
                else:
                        for x, y in validate_set:
                                predicted.append(self.feed_forward(x)[0][0])
                                label.append(y)
                        rmse = mt.Metrics().RootMeanSquareError(np.asarray(label), predicted)
                        #print('Epoch {0} completed with rmse::: {1}'.format(epoch, rmse))
                        return rmse
                
                