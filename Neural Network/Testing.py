from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

if __name__=='__main__':
     # pen =[]
     # car = []
     # for repeat in range(5):
     #      (trash, penAccuracy) = testPenData()
     #      (trash, carrAccuracy) = testCarData()
     #      pen.append(penAccuracy)
     #      car.append(carrAccuracy)
     pen = [0.836764, 0.789308, 0.769011, 0.804746, 0.785020]
     car = [0.680628, 0.626309, 0.702880, 0.687173, 0.727094]
     max(pen)
     print "Pen max: %f"%max(pen)
     print "Pen Average: %f" % average(pen)
     print "Pen Standard Deviation: %f" % stDeviation(pen)
     print "Car max: %f" % max(car)
     print "Car Average: %f" % average(car)
     print "Car Standard Deviation: %f" % stDeviation(car)

     # pen = []
     # for repeat in range(5):
     #      (trash, penAccuracy) = testPenData([40])
     #      pen.append(penAccuracy)
     # print "With 40 hidden layers, we have"
     # print "Pen max: %f" % max(pen)
     # print "Pen Average: %f" % average(pen)
     # print "Pen Standard Deviation: %f" % stDeviation(pen)
     # print "\n"
