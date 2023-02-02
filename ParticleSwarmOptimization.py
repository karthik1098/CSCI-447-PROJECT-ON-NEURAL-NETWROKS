
import Particle
import numpy as np
from sklearn.model_selection import train_test_split
import Metrics as mt

class ParticleSwarmOptimization:
    
        def __init__(self, trainSet, isClassification, networkStructure):
            # initialize network structure and layers
            self.isClassification = isClassification
            self.networkStructure = networkStructure
            self.trainSet, self.valSet = train_test_split(trainSet, test_size = 0.2, random_state = 0)
            self.outputLayer = []
            for i in range(networkStructure[-1]):
                self.outputLayer.append(0)
            
            # drop class column
            self.trainClass = np.array(self.trainSet[self.trainSet.columns[-1]])
            self.trainSet = self.trainSet.drop([self.trainSet.columns[-1]], axis = 'columns')
            self.valClass = np.array(self.valSet[self.valSet.columns[-1]])
            self.valSet = self.valSet.drop([self.valSet.columns[-1]], axis = 'columns')
            
            # initilize PSO attributes
            self.swarm_size = 100
            self.swarm = []
            self.gbest_fitness = float('-inf')
            self.gbest_position = []
            self.initSwarm()
            
        # intialize the population of individuals
        def initSwarm(self):
            for i in range(self.swarm_size):
                new_particle = Particle.Particle(self.networkStructure, self.isClassification)
                new_particle.setFitness(self.trainSet, self.trainClass)
                new_particle.best_fitness = new_particle.fitness
                fitness = new_particle.best_fitness
                if(new_particle.best_fitness > self.gbest_fitness):
                    self.gbest_fitness = new_particle.best_fitness
                    self.gbest_position = new_particle.best_position
                self.swarm.append(new_particle)
            swarm = self.swarm
            print('finished swarm init')
                
        def train(self):
            maxIterations = 5
            curIteration = 0
            iterration_list = []
            best_fitnesses = []
            
            while(curIteration < maxIterations):
                for particle in self.swarm:
                    gbest_fitness = self.gbest_fitness
                    self.gbest_fitness, self.gbest_position = particle.move(self.gbest_fitness, self.gbest_position, self.trainSet, self.trainClass)
                iterration_list.append(curIteration)
                if not self.isClassification: gbest_fitness = -self.gbest_fitness
                best_fitnesses.append(gbest_fitness)
                print(str.format("Epoch: {0} :: Global Best Fitness: {1} ", curIteration, round(gbest_fitness, 5)))
                curIteration += 1
                
            return iterration_list, best_fitnesses
        
        def test(self, testSet, testClass):
            predicted = self.predict(testSet)
            fitness = self.evalFitness(predicted, testClass)
            if not self.isClassification: fitness = -fitness
            print("most fit training fitness", fitness)
            return predicted
            
        def predict(self, testSet):
            predictions = []
            for index, point in testSet.iterrows():
                output = self.feedforward(point.values)
                predicted = self.predictedOutput(output)
                predictions.append(predicted)
            return predictions
        
        def predictedOutput(self, output):
            # get prediction for classification vs regression
            if(self.isClassification):
                #print(output)
                predicted = np.argmax(output)
                #print(predicted)
            else:
                #predicted = output[0][0]
                predicted = output[0]
            #print(predicted)
            return predicted
        
        # feedforward to get output of network
        def feedforward(self, trainPoint):
            current = trainPoint
            #print(current)
            current = np.append(current, [1])
            for i in range(len(self.networkStructure)-1):
                current = np.dot(current, self.gbest_position[i])
                current = mt.Metrics().sigmoid(current)
                if i != len(self.networkStructure)-2:
                    current = np.append(current, [1])
            return current
        
        def evalFitness(self, predictions, expected):
            fitness = 0
            if self.isClassification:
                accuracy = 0
                for i in range(len(predictions)):
                    if predictions[i] == expected[i]:
                        accuracy += 1
                fitness = accuracy/len(predictions)
            else:
                loss = 0
                for i in range(len(predictions)):
                    loss += mt.Metrics().RootMeanSquareError(expected[i], predictions[i])
                fitness = -loss/len(predictions)
            return fitness
                    
                
                
        
