import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        
        return 1 / (1 + exp(-value))
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as rounded g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The rounded value of the sigmoid of the weighted input
        """
        
        inActs.insert(0, float(1))
        totalInput = 0
        for index in range(len(inActs)):
		totalInput += self.weights[index] * inActs[index]
        return round(self.sigmoid(totalInput), 0)

    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        
        return self.sigmoid(value) * (1 - self.sigmoid(value))
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        
        inActs.insert(0, 1)
        totalInput = 0
        for index in range(len(inActs)):
		totalInput += self.weights[index] * inActs[index]
        return self.sigmoidDeriv(totalInput)
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        
        inActsCopy = copy.deepcopy(inActs)
        inActsCopy.insert(0, 1)
        for index in range(len(self.weights)):
             difference = alpha * delta * inActsCopy[index]
             self.weights[index] += difference
             totalModification += abs(difference)

        return totalModification

    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        
        finalList = []
        finalList.append(inActs)

        for layer in self.layers:
             newInActs = []
             for percep in layer:
                  newInActs.append(percep.sigmoidActivation(copy.deepcopy(inActs)))
             inActs = newInActs
             finalList.append(inActs)

        return finalList

    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        
        Args: 
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons, 
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """
        #keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0
        summedError = 0
        inputSizes = 0

        for example in examples:#for each example
            deltas = {}#keep track of deltas to use in weight change
            
            
            """Get output of all layers"""
            
            """
            Calculate output errors for each output perceptron and keep track 
            of error sum. Add error delta values to list.
            """
            
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            Be careful to account for bias inputs! 
            """
            
            """
            Having aggregated all deltas, update the weights of the 
            hidden and output layers accordingly.
            """

            #delta's will be stored as tuples like: ((layer, index within layer), delta)

            #outputList represents the output generated from feed forward
            outputList = self.feedForward(example[0])

            #realOuterLayer is the classification from the example
            realOuterLayer = example[1]

            #this is the output of the layer before the last layer
            outputBeforeLastOutput = outputList[len(outputList) - 2]

            #temporarily stores the deltas
            tempDeltaList = []
            for index in range(len(realOuterLayer)):
                 difference = realOuterLayer[index] - outputList[len(outputList) - 1][index]
                 gPrimeJ = self.outputLayer[index].sigmoidActivationDeriv(copy.deepcopy(outputBeforeLastOutput))

                 result = difference * gPrimeJ
                 summedError += (difference * difference) / 2

                 deltas[(len(outputList) - 2, index)] = result
                 tempDeltaList.insert(index, result)

            #We've now calculated the deltas for the output layer
            reversedList = (range(len(self.layers) - 1))
            reversedList.reverse()
            for layerIndex in reversedList:
                 layer = self.layers[layerIndex]
                 #when we're at the layer that takes in the original input
                 if not layerIndex:
                      for percepInd in range(len(layer)):
                           percep = layer[percepInd]
                           gPrimeJ = percep.sigmoidActivationDeriv(copy.deepcopy(example[0]))
                           difference = self.getSummedOutputWeightsTimesDeltas(layerIndex, percepInd, tempDeltaList)
                           result = difference * gPrimeJ
                           deltas[(layerIndex, percepInd)] = result

            #update weights and tally average weight change              
            for layerIndex in range(len(self.layers)):
                 layer = self.layers[layerIndex]
                 for percepInd in range(len(layer)):
                      node = layer[percepInd]
                      delta = deltas[(layerIndex, percepInd)]
                      if not layerIndex:
                         weightChange = node.updateWeights(example[0], alpha, delta)
                         averageWeightChange += weightChange
                         inputSizes += (len(example[0]) + 1)
                      else:
                         averageWeightChange += node.updateWeights(outputList[layerIndex], alpha, delta)
                         inputSizes += (len(outputList[layerIndex]) + 1)   


        #end for each example
        averageError = summedError / (len(examples) * len(self.outputLayer))
        averageWeightChange = averageWeightChange / inputSizes
        """Calculate final output"""
        return (averageError, averageWeightChange)

    def getSummedOutputWeightsTimesDeltas(self, currentLayer, currentPeceptron, deltaList):
          output = 0
          for nodeInd in range(len(self.layers[currentLayer + 1])):
               node = self.layers[currentLayer + 1][nodeInd]
               output += (node.weights[currentPeceptron + 1] * deltaList[nodeInd])
          return output


def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration=0
    trainError=0
    weightMod=sys.maxint
    
    """
    Iterate for as long as it takes to reach weight modification threshold
    """
       
        
    while (iteration < maxItr and weightMod > weightChangeThreshold):
         trainError, weightMod = nnet.backPropLearning(examples[0], alpha)
         iteration += 1
         time = datetime.now().time()
         # print "On iteration %d at time %s; training error %f and weight change %f"%(iteration,str(time),trainError,weightMod)
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
    """ 
    
    testError = 0
    testGood = 0
    for example in examplesTest:
         result = nnet.feedForward(example[0])
         if result[len(result) - 1] == example[1]:
              testGood += 1
         else:
              testError += 1
    
    testAccuracy= float(testGood) / (testGood + testError)
    
    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    
    """return something"""
    return (nnet, testAccuracy)

