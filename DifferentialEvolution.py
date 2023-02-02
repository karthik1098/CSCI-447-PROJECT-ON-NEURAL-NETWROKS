import Individual
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
import copy
class DE:
        def __init__(self, trainSet, isClassification, networkStructure):
            # initialize network structure and layers
            self.beta = 0.5
            self.pr = 0.5
            self.classification = isClassification
            self.networkStructure = networkStructure
            self.trainSet, self.valSet = train_test_split(trainSet, test_size = 0.3)
            self.outputLayer = []
            for i in range(networkStructure[-1]):
                self.outputLayer.append(0)
            
            # drop class column
            self.trainClass = np.array(self.trainSet[self.trainSet.columns[-1]])
            self.trainSet = self.trainSet.drop([self.trainSet.columns[-1]], axis = 'columns')
            self.valClass = np.array(self.valSet[self.valSet.columns[-1]])
            self.valSet = self.valSet.drop([self.valSet.columns[-1]], axis = 'columns')
            
            # initilize DE attributes
            self.popSize = 100
            self.population = []
            self.bestIndivdiual = -1
            self.bestIndivdiualFitness = -1
            self.performance = []
            self.initPop()

        # intialize the population of individuals
        def initPop(self):
            for i in range(self.popSize):
                newInd = Individual.Individual(self.networkStructure, self.classification)
                newInd.setFitness(self.trainSet, self.trainClass)
                # find best fit
                if newInd.fitness > self.bestIndivdiualFitness:
                    self.bestIndivdiual = copy.deepcopy(newInd)
                    self.bestIndivdiualFitness = newInd.fitness
                # add individual to population
                self.population.append(newInd)
    
        # train algorithm
        def train(self):
            maxIterations = 70
            generation = 0
            generationList = range(0, maxIterations)
            
            # repeat until max iterations is reached
            while(generation < maxIterations):
                print('generation: ', generation)
                for x in range(len(self.population)):
                        pop = self.population[x]
                        #do mutation
                        donor_v = self.mutation(x)
                        #crossover
                        crossover_p = rn.uniform(0, 1)
                        if crossover_p > self.pr:
                                new_pop = donor_v
                        else:
                                new_pop = pop
                        if new_pop.fitness > pop.fitness:
                               self.population[x] = new_pop   
                               if self.bestIndivdiual.fitness < new_pop.fitness:
                                       self.bestIndivdiual = new_pop
                                       self.bestIndivdiual.fitness = new_pop.fitness
                 # monitor most fit training fitness
                print("----training fitness----")
                #overallBestFitness = self.bestIndivdiualFitness
                #if not self.classification: overallBestFitness = -overallBestFitness
                #print("overall best training fitness", round(overallBestFitness, 5))
                currentBestFitness = self.population[self.getMostFit(range(self.popSize))].fitness
                if not self.classification: currentBestFitness = -currentBestFitness
                print("current best training fitness", round(currentBestFitness, 5))
                                
                # monitor validation fitness
                print("----validation fitness----")
                self.test(self.valSet, self.valClass)    
                generation += 1
            return generationList, self.performance        
        # get fittest of selected individuals
        def getMostFit(self, selected):
            maxFitness = -1
            mostFitIndex = -1
            for i in selected:
                fitness = self.population[i].fitness
                if(fitness > maxFitness):
                    maxFitness = fitness
                    mostFitIndex = i
            return mostFitIndex
        

                       
        # randomly select some individuals to mutate genes
        def mutation(self, pos):
                x = self.population[pos].chromosome
                #randomly select three population for the mutation
                random_index = rn.sample(range(len(self.population)), 3) 
                x1 = self.population[random_index[0]].chromosome
                x2 = self.population[random_index[1]].chromosome
                x3 = self.population[random_index[2]].chromosome
                x_arr = [np.array(z) for z in x]
                x1_arr = [np.array(z) for z in x1]
                x2_arr = [np.array(z) for z in x2]
                x3_arr = [np.array(z) for z in x3]
                #apply the mutation function
                for i in range(len(x_arr)):
                        x_arr[i] = x1_arr[i] + self.beta * (x2_arr[i] - x3_arr[i])
                mutated_chromosome = [z.tolist() for z in x_arr]
                #created a mutated population to check fitness
                mutated_pop = Individual.Individual(self.networkStructure, self.classification)
                mutated_pop.setChoromosome(mutated_chromosome)
                mutated_pop.setFitness(self.trainSet, self.trainClass)
                return mutated_pop
        
        # predict based on most fit individual
        def test(self, testSet, testClass):
            
            # test with overall most fit
            bestPredicted = self.predict(testSet, self.bestIndivdiual)
            bestFitness = self.bestIndivdiual.evalFitness(bestPredicted, testClass)
            bestFitnessP = bestFitness
            if not self.classification: bestFitnessP = -bestFitness
            print("overall most fit test fitness", round(bestFitnessP, 5))

            # test with current most fit
            mostFitIndex = self.getMostFit(range(self.popSize))
            mostFit = self.population[mostFitIndex]
            predicted = self.predict(testSet, mostFit)
            fitness = mostFit.evalFitness(predicted, testClass)
            fitnessP = fitness
            if not self.classification: fitnessP = -fitness
            print("current most fit test fitness", round(fitnessP, 5), "\n")
            # store fitness to return
            self.performance.append(fitnessP)
            
            # if current fitness is better than previous, store this indivdiual
            if fitness > bestFitness:
                self.bestIndivdiual = copy.deepcopy(mostFit)
                self.bestIndivdiualFitness = fitness
                bestPredicted = predicted

            return bestPredicted
        
        # predict based on individual's chromosome
        def predict(self, testSet, individual):
            predictions = []
            # use each test point and the selected individual
            for index, point in testSet.iterrows():
                output = individual.feedforward(point.values)
                predicted = individual.predict(output)
                predictions.append(predicted)
            return predictions
        
            