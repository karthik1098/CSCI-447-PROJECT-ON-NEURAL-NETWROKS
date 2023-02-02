import numpy as np
import random as rn
import Metrics as mt

class Particle:
    
    def __init__(self, particleStructure, isClassification):
        self.isClassification = isClassification
        self.particleStructure = particleStructure
        # multipliers for positions and velocity 
        self.W = 0.5
        self.c1 = 0.8
        self.c2 = 0.9
        self.position = []
        self.initializePosition()
        self.best_position = self.position
        best_position = self.best_position
        
        self.velocity = []
        self.initializeVelocity()
        
        self.fitness = 0
        self.best_fitness = 0          
        
    # initialize position randomly
    def initializePosition(self):
        for i in range(1, len(self.particleStructure)):
            numRows = self.particleStructure[i-1] + 1
            numCols = self.particleStructure[i]
            initPos = np.random.normal(0, self.c1, (numRows, numCols))
            self.position.append(initPos.tolist())
            length = numRows*numCols
            #self.positionLength += length            
    
    # initialize velocity randomly
    def initializeVelocity(self):
        for i in range(1, len(self.particleStructure)):
            numRows = self.particleStructure[i-1] + 1
            numCols = self.particleStructure[i]
            initVel = np.random.normal(0, self.W, (numRows, numCols))
            self.velocity.append(initVel.tolist())
    
    # move the particle 
    def move(self, gbest_fitness, gbest_position, trainSet, trainClass):
        self.updateVelocity(gbest_position)
        #self.velocity = (self.W*self.velocity) + (self.c1*rn.random()) * (self.best_position - self.position) + (rn.random()*self.c2) * (gbest_position - self.position)
        #self.position = self.position + self.velocity
        self.updatePosition()
        self.setFitness(trainSet, trainClass)
        # fitness is defined as accuracy for classification and -RMSE for regression, so a larger value is better
        fitness = self.fitness
        gbest = gbest_fitness
        if(self.fitness > gbest_fitness):
            gbest_fitness = self.fitness
            gbest_position = self.position
            self.best_fitness = self.fitness
            self.best_position = self.position
        elif(self.fitness > self.best_fitness):
            self.best_fitness = self.fitness
            self.best_position = self.position
        return gbest_fitness, gbest_position
    
    def updateVelocity(self, gbest_position):
        r1 = rn.random() * self.c1
        r2 = rn.random() * self.c2
        for i in range(len(self.position)):
            for j in range(len(self.position[i])):
                for k in range(len(self.position[i][j])):
                    self.velocity[i][j][k] = self.W * self.velocity[i][j][k] + r1 * (self.best_position[i][j][k] - self.position[i][j][k]) + r2 * (gbest_position[i][j][k] - self.position[i][j][k])
    
    def updatePosition(self):
        for i in range(len(self.position)):
            for j in range(len(self.position[i])):
                for k in range(len(self.position[i][j])):
                    self.position[i][j][k] = self.position[i][j][k] + self.velocity[i][j][k]
                    
    # calculate fitness for the individual
    def setFitness(self, trainSet, trainClass):
        # feedforward and get predictions
        predictions = []
        for index, point in trainSet.iterrows():
            output = self.feedforward(point.values)
            predicted = self.predict(output)
            predictions.append(predicted)
        self.fitness = self.evalFitness(predictions, trainClass)
    
    # feedforward to get output of network
    def feedforward(self, trainPoint):
        current = trainPoint
        #print(current)
        current = np.append(current, [1])
        #print(current)
        # feedforward through layers using positions as weights
        position = self.position
        for i in range(len(self.particleStructure)-1):
            current = np.dot(current, self.position[i])
            current = mt.Metrics().sigmoid(current)
            if i != len(self.particleStructure)-2:
                current = np.append(current, [1])
        return current
        
    def predict(self, output):
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
    
    # evaluate fitness
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