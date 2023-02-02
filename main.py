import LoadDataset as ld
import MLP
import GeneticAlg
import DifferentialEvolution as DE
import ParticleSwarmOptimization as PSO
import Metrics
import pandas as pd
import numpy as np
from graphs import plot_graph

class main:
        
        def __init__(self):
                #define variable to use inside class which may need tuning
                self.layers = [2]
                self.alldataset = ld.LoadDataset().load_data()          #load all dataset
                #check if dataset is classification
                self.IsClassificationDict = ld.LoadDataset().IsClassificationDict() 
                #define dataframe to store all the results
                self.allresults = pd.DataFrame(columns=['dataset', 'isClassification', 'hiddenLayers', 'method',
                                                        'accuracy', 'precision', 'recall', 'NRMSE'])
                self.algorithms = ["MLP", "GA", "DE", "PSO"]
                self.algorithms = ["GA"]
        
        def main(self):
                # for each dataset call each algorithm and number of hidden layers
                for dataset in self.alldataset:         
                        print('current dataset ::: {0} \n'.format(dataset))
                        data = self.alldataset.get(dataset)
                        isClassification = self.IsClassificationDict.get(dataset)
                        trainset, testset = self.testtrainsplit(data)
                        len_input_neuron = len(data.columns[:-1])
                        len_output_neuron = 1
                        if isClassification: len_output_neuron = len(data[data.columns[-1]].unique())
                        
                        # run algorithms with h hidden layer and store the evaluated performance.
                        for i in range(len(self.layers)):
                                h = self.layers[i]
                                structure = self.get_network_structure(dataset, len_input_neuron, len_output_neuron,  h)
                                
                                # call each algorithm
                                for j in range(len(self.algorithms)):
                                    if self.algorithms[j] == "MLP":
                                        predicted, label = self.run_MLP(dataset, data, trainset, testset, isClassification, structure, h)
                                        self.performance_measure(predicted, label, dataset, isClassification, h, self.algorithms[j])
                                        
                                    elif self.algorithms[j] == "GA":
                                        predicted, label = self.run_GA(dataset, trainset, testset, isClassification, structure, h)
                                        self.performance_measure(predicted, label, dataset, isClassification, h, self.algorithms[j])
                                    
                                    elif self.algorithms[j] == "DE":
                                        predicted, label = self.run_DE(dataset, trainset, testset, isClassification, structure, h)
                                        self.performance_measure(predicted, label, dataset, isClassification, h, self.algorithms[j])    
                                        
                                    elif self.algorithms[j] == "PSO":
                                        predicted, label = self.run_PSO(dataset, trainset, testset, isClassification, structure, h)
                                        self.performance_measure(predicted, label, dataset, isClassification, h, self.algorithms[j])
                return self.allresults
                        
                        
        ## 10 Fold Cross Validation
        
        def testtrainsplit(self, data):
                data = data.sample(frac=1)      
                folds = np.array_split(data, 10)
                
                test_set = 0
                for y in range(len(folds)):
                    train = pd.DataFrame()
                    
                    for x in range(len(folds)):
                        if x == test_set:
                            testset = folds[x]
                        else:
                            train = train.append(folds[x])
                    
                    
                    
                            
                testset = testset
                trainset = train
                return trainset, testset
        
        def run_MLP(self,key, dataset, train, test, isClassification, structure, num_hidden_layers):
                #takes network architecture arguments and run Class MLP accordingly                
                x_train, train_label = ld.LoadDataset().get_neural_net_input_shape(dataset, train, isClassification)
                x_test, test_label = ld.LoadDataset().get_neural_net_input_shape(dataset, test, isClassification)
                #call class MLP based on the network structure provided
                net = MLP.MLP(structure)
                epoc, result = net.train(x_train, train_label, 400, 30, 3, isClassification)
                plot_graph(key + ' with ' + str(num_hidden_layers) + ' hidden layers', epoc, result)
                predicted = net.test(x_test, isClassification)
                return predicted, test.iloc[:, -1]
            
        def run_GA(self, key, train, test, isClassification, structure, num_hidden_layers):
                # train and plot convergence
                GA = GeneticAlg.GeneticAlg(train, isClassification, structure)
                epoc, result = GA.train()
                plot_graph(key + ' with ' + str(num_hidden_layers) + ' hidden layers', epoc, result)
                
                # drop class variable
                testClass = test[test.columns[-1]]
                testSet = test.drop([test.columns[-1]], axis = 'columns')
                
                # get predictions
                predicted = GA.test(testSet, np.array(testClass))
                return predicted, testClass
        
        def run_DE(self, key, train, test, isClassification, structure, num_hidden_layers):
                # train and plot convergence
                de = DE.DE(train, isClassification, structure)
                epoc, result = de.train()
                #plot_graph(key + ' with ' + str(num_hidden_layers) + ' hidden layers', epoc, result)
                
                # drop class variable
                testClass = test[test.columns[-1]]
                testSet = test.drop([test.columns[-1]], axis = 'columns')
                
                # get predictions
                predicted = de.test(testSet, np.array(testClass))
                return predicted, testClass
            
        def run_PSO(self, key, train, test, isClassification, structure, num_hidden_layers):
                pso = PSO.ParticleSwarmOptimization(train, isClassification, structure)
                epoc, result = pso.train()
                #plot_graph(key + ' with ' + str(num_hidden_layers) + ' hidden layers', epoc, result)
                # drop class variable
                testClass = test[test.columns[-1]]
                testSet = test.drop([test.columns[-1]], axis = 'columns')
                
                # get predictions
                predicted = pso.test(testSet, np.array(testClass))
                return predicted, testClass
                
        def get_network_structure(self, key, input_neuron, output_neuron, num_hidden_layers):
                if num_hidden_layers == 0:
                        structure = [input_neuron, output_neuron]
                elif num_hidden_layers == 1:
                        hidden_layer = ld.LoadDataset().get1sthiddenlayernode(key)
                        structure = [input_neuron, hidden_layer, output_neuron]
                else:
                        hidden_layer_list = ld.LoadDataset().get2ndhiddenlayernode(key)
                        structure = [input_neuron, hidden_layer_list[0], hidden_layer_list[1], output_neuron]
                return structure
                        
        def performance_measure(self, predicted, labels, dataset, isClassification, h, method):
                #evaluate confusion metrix or root mean square error based on dtaset
                mtrx = Metrics.Metrics()
                if (isClassification):
                        acc, prec, recall = mtrx.confusion_matrix(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, acc, prec, recall, 0)
                         
                else:
                        rmse = mtrx.RootMeanSquareError(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, 0, 0, 0, rmse)
        
        def update_result(self, dataset, isClassification, h, method, acc, prec, recall, rmse):
                #store result in a dataframe.
                self.allresults = self.allresults.append({'dataset': dataset, 'isClassification': isClassification,
                                                'hiddenLayers': h, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'NRMSE': rmse}, ignore_index=True)
        
results = main().main()
results.to_csv('results.csv')
                