from math import log
import math
from sys import maxint
import copy

class Node:
  """
  A simple node class to build our tree with. It has the following:
  
  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by. 
  islead (boolean): whether this is a leaf. False.
  """
  
  def __init__(self,attr):
	self.children = {}
	self.attr = attr
	self.isleaf = False

class LeafNode(Node):
	"""
	A basic extension of the Node class with just a value.
	
	value (str): Since this is a leaf node, a final value for the label.
	islead (boolean): whether this is a leaf. True.
	"""
	def __init__(self,value):
		self.value = value
		self.isleaf = True
	
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
	self.root = root

  def prettyPrint(self):
	print str(self)
	
  def preorder(self,depth,node):
	if node is None:
	  return '|---'*depth+str(None)+'\n'
	if node.isleaf:
	  return '|---'*depth+str(node.value)+'\n'
	string = ''
	for val in node.children.keys():
	  childStr = '|---'*depth
	  childStr += '%s = %s'%(str(node.attr),str(val))
	  string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
	return string    

  def count(self,node=None):
	if node is None:
	  node = self.root
	if node.isleaf:
	  return 1
	count = 1
	for child in node.children.values():
	  if child is not None:
		count+= self.count(child)
	return count  

  def __str__(self):
	return self.preorder(0, self.root)
  
  def classify(self, classificationData):
	"""
	Uses the classification tree with the passed in classificationData.`
	
	Args:
		classificationData (dictionary<string,string>): dictionary of attribute values
	Returns:
		str
		The classification made with this tree.
	"""
	
	node = self.root
  	while not node.isleaf:
		node = node.children[classificationData[node.attr]]
	return node.value
  
def getPertinentExamples(examples,attrName,attrValue):
	"""
	Helper function to get a subset of a set of examples for a particular assignment 
	of a single attribute. That is, this gets the list of examples that have the value 
	attrValue for the attribute with the name attrName.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrName (str): the name of the attribute to get counts for
		attrValue (str): a value of the attribute
		className (str): the name of the class
	Returns:
		list<dictionary<str,str>>
		The new list of examples.
	"""
	newExamples = []
	
	for example in examples:
		if example[attrName] == attrValue:
			newExamples.append(example)
	return newExamples
  
def getClassCounts(examples,className):
	"""
	Helper function to get a list of counts of different class values
	in a set of examples. That is, this returns a list where each index 
	in the list corresponds to a possible value of the class and the value
	at that index corresponds to how many times that value of the class 
	occurs.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		className (str): the name of the class
	Returns:
		dictionary<string,int>
		This is a dictionary that for each value of the class has the count
		of that class value in the examples. That is, it maps the class value
		to its count.
	"""
	classCounts = {}
	
	for example in examples:
		value = example[className]
		if value in classCounts.keys():
			classCounts[value] += 1
		else:
			classCounts[value] = 1
	return classCounts

def getMostCommonClass(examples,className):
	"""
	A freebie function useful later in makeSubtrees. Gets the most common class
	in the examples. See parameters in getClassCounts.
	"""
	counts = getClassCounts(examples,className)
	return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples,attrName,attrValues,className):
	"""
	Helper function to get a list of counts of different class values
	corresponding to every possible assignment of the passed in attribute. 
	  That is, this returns a list of lists, where each index in the list 
	  corresponds to an assignment of the attribute named attrName and holds
	  the counts of different class values for the subset of the examples
	  that have that assignment of that attribute.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrName (str): the name of the attribute to get counts for
		attrValues (list<str>): list of possible values for attribute
		className (str): the name of the class
	Returns:
		list<list<int>>
		This is a list that for each value of the attribute has a
		list of counts of class values. No specific ordering of the
		classes in each list is needed.
	"""
	counts=[]
	
	for value in attrValues:
		classSeen = []
		countSeen = []
		for example in examples:
			if example[attrName] == value and example[className] not in classSeen:
				classSeen.append(example[className])
				countSeen.append(1)
			elif example[attrName] == value and example[className] in classSeen:
				countSeen[classSeen.index(example[className])] += 1
		counts.append(countSeen)

	return counts
		

def setEntropy(classCounts):
	"""
	Calculates the set entropy value for the given list of class counts.
	This is called H in the book. Note that our labels are not binary,
	so the equations in the book need to be modified accordingly. Note
	that H is written in terms of B, and B is written with the assumption 
	of a binary value. B can easily be modified for a non binary class
	by writing it as a summation over a list of ratios, which is what
	you need to implement.
	
	Args:
		classCounts (list<int>): list of counts of each class value
	Returns:
		float
		The set entropy score of this list of class value counts.
	"""
	
	for index in range(len(classCounts)):
		if not classCounts[index]:
			classCounts[index] = 0

	listSum = sum(classCounts)
	ratioList = []
	for classCount in classCounts:
		x = float(classCount) / listSum
		ratioList.append(x)
	entropy = 0
	for ratio in ratioList:
		entropy += -ratio * log(ratio, 2)
	return entropy
   

def remainder(examples,attrName,attrValues,className):
	"""
	Calculates the remainder value for given attribute and set of examples.
	See the book for the meaning of the remainder in the context of info 
	gain.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrName (str): the name of the attribute to get remainder for
		attrValues (list<string>): list of possible values for attribute
		className (str): the name of the class
	Returns:
		float
		The remainder score of this value assignment of the attribute.
	"""
	
	classCounts = getAttributeCounts(examples,attrName,attrValues,className)
	remainder = 0
	totalSum = 0
	for attr in classCounts:
		if attr:
			totalSum += sum(attr)
	for attrCount in classCounts:
		countSum = float(sum(attrCount))
		remainder += countSum / totalSum * setEntropy(attrCount)
	return remainder


		  
def infoGain(examples,attrName,attrValues,className):
	"""
	Calculates the info gain value for given attribute and set of examples.
	See the book for the equation - it's a combination of setEntropy and
	remainder (setEntropy replaces B as it is used in the book).
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrName (str): the name of the attribute to get remainder for
		attrValues (list<string>): list of possible values for attribute
		className (str): the name of the class
	Returns:
		float
		The gain score of this value assignment of the attribute.
	"""
	
	classCount = getClassCounts(examples,className)
	ratios = classCount.values()
	entropy = setEntropy(ratios)

	remainderValue = remainder(examples,attrName,attrValues,className)

	return entropy - remainderValue
  
def giniIndex(classCounts):
	"""
	Calculates the gini value for the given list of class counts.
	See equation in instructions.
	
	Args:
		classCounts (list<int>): list of counts of each class value
	Returns:
		float
		The gini score of this list of class value counts.
	"""
	
	giniValue = 0
	sumList = sum(classCounts)
	for value in classCounts:
		giniValue += math.pow(float(value) / sumList, 2)

	return 1 - giniValue
  
def giniGain(examples,attrName,attrValues,className):
	"""
	Return the inverse of the giniD function described in the instructions.
	The inverse is returned so as to have the highest value correspond 
	to the highest information gain as in entropyGain. If the sum is 0,
	return sys.maxint.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrName (str): the name of the attribute to get counts for
		attrValues (list<string>): list of possible values for attribute
		className (str): the name of the class
	Returns:
		float
		The summed gini index score of this list of class value counts.
	"""
	
	classCounts = getAttributeCounts(examples,attrName,attrValues,className)
	giniValue = 0
	totalSum = 0
	for attr in classCounts:
		if attr:
			totalSum += sum(attr)
	for attrCount in classCounts:
		countSum = float(sum(attrCount))
		temp = giniIndex(attrCount)
		giniValue += countSum / totalSum * giniIndex(attrCount)

	if not giniValue:
		return  maxint
	return math.pow(giniValue, -1)


	
def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
	"""
	Creates the classification tree for the given examples. Note that this is implemented - you
	just need to imeplement makeSubtrees.
	
	Args:
		examples (list<dictionary<str,str>>): list of examples
		attrValues (dictionary<string,list<string>>): list of possible values for attribute
		className (str): the name of the class
		classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
		gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
	Returns:
		Tree
		The classification tree for this set of examples
	"""
	remainingAttributes=attrValues.keys()
	return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,setScoreFunc,gainFunc))
	
def makeSubtrees(remainingAttributes,examples,attributeValues,className,setScoreFunc,gainFunc):
	"""
	Creates a classification tree Node and all its children. This returns a Node, which is the root
	Node of the tree constructed from the passed in parameters. This should be implemented recursively,
	and handle base cases for zero examples or remainingAttributes as covered in the book.    

	Args:
		remainingAttributes (list<string>): the names of attributes still not used
		examples (list<dictionary<str,str>>): list of examples
		attrValues (dictionary<string,list<string>>): list of possible values for attribute
		className (str): the name of the class
		setScoreFunc (func): the function to score classes (ie classEntropy or gini)
		gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
		children (dictionary<str,Node>): A mapping from attribute value to a child node
	Returns:
		Node or LeafNode
		The classification tree node optimal for the remaining set of attributes.
	"""
	

	if len(getClassCounts(examples, className)) == 1:
		#we've reached a node in which there's only one classification.  Hoorah.
		temp = examples[0][className]
		leafNode = LeafNode(temp)
		return leafNode
	elif not remainingAttributes:
		#We've reached a leaf node that has more than one classification, take the most common
		temp = getMostCommonClass(examples, className)
		leafNode = LeafNode(temp)
		return leafNode
	else:
		bestGain = 0
		bestAttr = remainingAttributes[0]

		#find the attribute with the highest entropy gain
		for attrName in remainingAttributes:
			gain = infoGain(examples, attrName, attributeValues[attrName], className)
			if gain > bestGain:
				bestGain = gain
				bestAttr = attrName

		#remove from list of attributes to look at		
		remainingAttributes.remove(bestAttr)
		node = Node(bestAttr)
		children = {}

		#go through each attribute value for the best attribute
		for attrValue in attributeValues[bestAttr]:
			#get the exampels associated with the best attribute
			newExamples = getPertinentExamples(examples, bestAttr, attrValue)
			if not newExamples:
				#if None, means we've reached the bottom of a tree
				newNode = LeafNode(getMostCommonClass(examples, className))
			else:
				#recurse through the tree with the new set of examples associated with the assignment of the values to the best attribute 
				newNode = makeSubtrees(copy.copy(remainingAttributes),newExamples,attributeValues,className,setScoreFunc,gainFunc)

			if newNode != None:
				children[attrValue] = newNode

		node.children = children

		if not node.children:
			node = LeafNode(bestAttr)
		return node