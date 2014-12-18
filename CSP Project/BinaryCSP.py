from collections import deque

"""
	Base class for unary constraints
	Implement isSatisfied in subclass to use
"""
class UnaryConstraint:
	def __init__(self, var):
		self.var = var

	def isSatisfied(self, value):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var


"""	
	Implementation of UnaryConstraint
	Satisfied if value does not match passed in paramater
"""
class BadValueConstraint(UnaryConstraint):
	def __init__(self, var, badValue):
		self.var = var
		self.badValue = badValue

	def isSatisfied(self, value):
		return not value == self.badValue

	def __repr__(self):
		return 'BadValueConstraint (%s) {badValue: %s}' % (str(self.var), str(self.badValue))


"""	
	Implementation of UnaryConstraint
	Satisfied if value matches passed in paramater
"""
class GoodValueConstraint(UnaryConstraint):
	def __init__(self, var, goodValue):
		self.var = var
		self.goodValue = goodValue

	def isSatisfied(self, value):
		return value == self.goodValue

	def __repr__(self):
		return 'GoodValueConstraint (%s) {goodValue: %s}' % (str(self.var), str(self.goodValue))


"""
	Base class for binary constraints
	Implement isSatisfied in subclass to use
"""
class BinaryConstraint:
	def __init__(self, var1, var2):
		self.var1 = var1
		self.var2 = var2

	def isSatisfied(self, value1, value2):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var1 or var == self.var2

	def otherVariable(self, var):
		if var == self.var1:
			return self.var2
		return self.var1


"""
	Implementation of BinaryConstraint
	Satisfied if both values assigned are different
"""
class NotEqualConstraint(BinaryConstraint):
	def isSatisfied(self, value1, value2):
		if value1 == value2:
			return False
		return True

	def __repr__(self):
	    return 'NotEqualConstraint (%s, %s)' % (str(self.var1), str(self.var2))


class ConstraintSatisfactionProblem:
	"""
	Structure of a constraint satisfaction problem.
	Variables and domains should be lists of equal length that have the same order.
	varDomains is a dictionary mapping variables to possible domains.

	Args:
		variables (list<string>): a list of variable names
		domains (list<set<value>>): a list of sets of domains for each variable
		binaryConstraints (list<BinaryConstraint>): a list of binary constraints to satisfy
		unaryConstraints (list<UnaryConstraint>): a list of unary constraints to satisfy
	"""
	def __init__(self, variables, domains, binaryConstraints = [], unaryConstraints = []):
		self.varDomains = {}
		for i in xrange(len(variables)):
			self.varDomains[variables[i]] = domains[i]
		self.binaryConstraints = binaryConstraints
		self.unaryConstraints = unaryConstraints

	def __repr__(self):
	    return '---Variable Domains\n%s---Binary Constraints\n%s---Unary Constraints\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + '\n' for e in self.binaryConstraints]), \
	        ''.join([str(e) + '\n' for e in self.unaryConstraints]))


class Assignment:
	"""
	Representation of a partial assignment.
	Has the same varDomains dictionary stucture as ConstraintSatisfactionProblem.
	Keeps a second dictionary from variables to assigned values, with None being no assignment.

	Args:
		csp (ConstraintSatisfactionProblem): the problem definition for this assignment
	"""
	def __init__(self, csp):
		self.varDomains = {}
		for var in csp.varDomains:
			self.varDomains[var] = set(csp.varDomains[var])
		self.assignedValues = { var: None for var in self.varDomains }

	"""
	Determines whether this variable has been assigned.

	Args:
		var (string): the variable to be checked if assigned
	Returns:
		boolean
		True if var is assigned, False otherwise
	"""
	def isAssigned(self, var):
		return self.assignedValues[var] != None

	"""
	Determines whether this problem has all variables assigned.

	Returns:
		boolean
		True if assignment is complete, False otherwise
	"""
	def isComplete(self):
		for var in self.assignedValues:
			if not self.isAssigned(var):
				return False
		return True

	"""
	Gets the solution in the form of a dictionary.

	Returns:
		dictionary<string, value>
		A map from variables to their assigned values. None if not complete.
	"""
	def extractSolution(self):
		if not self.isComplete():
			return None
		return self.assignedValues

	def __repr__(self):
	    return '---Variable Domains\n%s---Assigned Values\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + ':' + str(self.assignedValues[e]) + '\n' for e in self.assignedValues]))

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def enqueue(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def deque(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

####################################################################################################


"""
	Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
	Do not assign value to var. Only check if this value would be consistent or not.
	If the other variable for a constraint is not assigned, then the new value is consistent with the constraint.
	You do not have to consider unary constraints, as those have already been taken care of.

	Args:
		assignment (Assignment): the partial assignment
		csp (ConstraintSatisfactionProblem): the problem definition
		var (string): the variable that would be assigned
		value (value): the value that would be assigned to the variable
	Returns:
		boolean
		True if the value would be consistent with all currently assigned values, False otherwise
"""
def consistent(assignment, csp, var, value):
	domain = assignment.varDomains[var]

	#if the value isn't even in the domain of the variable, return false
	if value not in domain:
		return False

	#iterater through the list of constraints to see if it violates any
	constraints = csp.binaryConstraints
	for item in constraints:
		if item.var1 == var:
			#getting the value of the other variable
			otherValue = assignment.assignedValues[item.var2]
			#if the other variable hasn't been assigned a value, it's true by default
			if otherValue != None:
				status = item.isSatisfied(value, otherValue)
				if not status:
					return False

		if item.var2 == var:
			otherValue = assignment.assignedValues[item.var1]
			#if the other variable hasn't been assigned a value, it's true by default
			if otherValue != None:
				status = item.isSatisfied(otherValue, value)
				if not status:
					return False

	return True


"""
	Recursive backtracking algorithm.
	A new assignment should not be created. The assignment passed in should have its domains updated with inferences.
	In the case that a recursive call returns failure or a variable assignment is incorrect, the inferences made along
	the way should be reversed. See maintainArcConsistency and forwardChecking for the format of inferences.

	Examples of the functions to be passed in:
	orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
	selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
	inferenceMethod: noInferences, maintainArcConsistency, forwardChecking

	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
		orderValuesMethod (function<assignment, csp, variable> returns list<value>): a function to decide the next value to try
		selectVariableMethod (function<assignment, csp> returns variable): a function to decide which variable to assign next
		inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>): a function to specify what type of inferences to use
			InferenceMethod will return None if the assignment has no solution. Otherwise it will return a set of inferences made. (The set can be empty.)
	Returns:
		Assignment
		A completed and consistent assignment. None if no solution exists.
"""
def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
	if assignment.isComplete():
		return assignment

	#gives the variable and order to assign values	
	var = selectVariableMethod(assignment, csp)
	values_list = orderValues(assignment, csp, var)

	#iterates through the possible values to be assigned
	for value in values_list:
		oldValue = assignment.assignedValues[var]

		#if the values is consistent with the constraints, make the assignment.  If not, move on to the next value
		if consistent(assignment,csp, var, value):
			#make assignment
			assignment.assignedValues[var] = value

			#find inferences
			inferences = inferenceMethod(assignment, csp, var, value)

			#if inferences is none, you've reached the end.  
			if inferences != None:

				#add in thing for inferences
				result = recursiveBacktracking(assignment, csp, orderValuesMethod,  selectVariableMethod, inferenceMethod)
				if result != None:
					return result

				#update information accordingly
				assignment.assignedValues[var] = oldValue
				for update in inferences:
					var1 = update[0]
					value1 = update[1]
					assignment.varDomains[var1].add(value1)
	return None


"""
	Uses unary constraints to eleminate values from an assignment.

	Args:
		assignment (Assignment): a partial assignment to expand upon
		csp (ConstraintSatisfactionProblem): the problem definition
	Returns:
		Assignment
		An assignment with domains restricted by unary constraints. None if no solution exists.
"""
def eliminateUnaryConstraints(assignment, csp):
	domains = assignment.varDomains
	for var in domains:
		for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
			for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
				domains[var].remove(value)
				if len(domains[var]) == 0:
					# Failure due to invalid assignment
					return None
	return assignment


"""
	Trivial method for choosing the next variable to assign.
	Uses no heuristics.
"""
def chooseFirstVariable(assignment, csp):
	for var in csp.varDomains:
		if not assignment.isAssigned(var):
			return var


"""
	Selects the next variable to try to give a value to in an assignment.
	Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		the next variable to assign
"""
import sys
def minimumRemainingValuesHeuristic(assignment, csp):
	nextVar = None
	domains = assignment.varDomains
	minLength = sys.maxint
	minList = None
	tie = False

	#finds the variable(s) with the least remaining values
	for var in domains:
		if not assignment.isAssigned(var):
			currLength = len(domains[var])

			#updates the current smallest length
			if currLength < minLength:
				minLength = currLength
				minList = var
				tie = False

			#updates that there is a tie	
			elif currLength == minLength:
				minList = list(minList)
				minList.append(var)
				tie = True

	#If more than one item in minList, there's a tie.  Use degree heuristic.  len(minList) > 1
	if minList != None and tie:
		maxVar = None
		maxConstraints = 0
		constraints = csp.binaryConstraints

		#goes through each variable to find the greatest degree
		for var in minList:
			numConstraints = 0

			#finds how many constrains involve the variable
			for item in constraints:
				if item.var1 == var:
					numConstraints += 1
				elif item.var2 == var:
					numConstraints += 1

			#updates the current largest degree
			if numConstraints > maxConstraints:
				maxVar = var
				maxConstraints = numConstraints

		#keeps looking through the variables.  Stops when either the length of constrains is zero or no variable is left in the min list
		if len(constraints) != 0:
			minList = maxVar
		else:
			minList = minList.pop(0)

	if minList != None and len(minList) <= 0:
		print "problem"

	nextVar = minList
	return nextVar


"""
	Trivial method for ordering values to assign.
	Uses no heuristics.
"""
def orderValues(assignment, csp, var):
	return list(assignment.varDomains[var])


"""
	Creates an ordered list of the remaining values left for a given variable.
	Values should be attempted in the order returned.
	The least constraining value should be at the front of the list.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable to be assigned the values
	Returns:
		list<values>
		a list of the possible values ordered by the least constraining value heuristic
"""
def leastConstrainingValuesHeuristic(assignment, csp, var):
	values = list(assignment.varDomains[var])
	descendList = []
	storingDict = {}
	finalList = []

	#goes through all the values
	for value in values:
		#helper function that finds the number of values excluded by the assignemnt of this particular value
		currConstraint = findConstrainingValue(assignment, csp, var, value)

		#update the dictionary storing the correspoding constrating with the value
		storingDict[value] = currConstraint

		#if no value has yet been added the list
		if len(descendList) == 0:
			descendList.append(value)
		else:
			#keeps the list in descending order
			i = 0
			placeIsFound = True
			while placeIsFound and i < len(descendList):
				conValue = storingDict[descendList[i]]
				if conValue <= currConstraint:
					#keep going through the list
					i += 1
				else:
					#the proper place in the list has been found.  Insert it
					descendList.insert(i, value)
					placeIsFound = False
			#handles the case when the value is dead last in the list
			if placeIsFound:
				descendList.append(value)
	return descendList

def findConstrainingValue(assignment, csp, var, value):
	#counter represents the number of values excluded by the assignment.  The smaller the better
	counter = 0
	constraints = csp.binaryConstraints
	for item in constraints:
		if item.var1 == var:
			#getting the value of the other variable
			for otherValue in assignment.varDomains[item.var2]:
				#if the other variable hasn't been assigned a value, it's true by default
				if otherValue != None:
					status = item.isSatisfied(otherValue, value)
					if not status:
						counter += 1
		if item.var2 == var:
			for otherValue in assignment.varDomains[item.var1]:
				otherValue = assignment.assignedValues[item.var1]
				#if the other variable hasn't been assigned a value, it's true by default
				if otherValue != None:
					status = item.isSatisfied(otherValue, value)
					if not status:
						counter += 1

	return counter

"""
	Trivial method for making no inferences.
"""
def noInferences(assignment, csp, var, value):
	return set([])


"""
	Implements the forward checking algorithm.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the function.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""
def forwardChecking(assignment, csp, var, value):
	inferences = set([])
	domains = assignment.varDomains

	constraints = csp.binaryConstraints

	#goes through constraint
	for item in constraints:
		if item.var1 == var:
			#getting the value of the other variable
			for otherValue in domains[item.var2]:
				status = item.isSatisfied(otherValue, value)

				#if not satisfied, add to list of inferences
				if not status:
					newInference = (item.var2, otherValue)
					inferences.add(newInference)

		#does the mirror operation except with var2 being the variable of interest			
		if item.var2 == var:
			for otherValue in domains[item.var1]:
				status = item.isSatisfied(otherValue, value)
				if not status:
					newInference = (item.var1, otherValue)
					inferences.add(newInference)

	#updates the domain according to the inferences				
	incomplete = False
	for update in inferences:
		var1 = update[0]
		value1 = update[1]
		domains[var1].remove(value1)
		if len(domains[var1]) == 0:
			incomplete = True

	#if the domain of any the variables is not exhausted, then an inconsistency is found, and None is returned
	if incomplete:
		#inferences are undone
		for update in inferences:
			var1 = update[0]
			value1 = update[1]
			domains[var1].add(value1)
		return None

	return inferences


"""
	Helper function to maintainArcConsistency and AC3.
	Remove values from var2 domain if constraint cannot be satisfied.
	Each inference should take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
	inferences made should be reversed before ending the fuction.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var1 (string): the variable with consistent values
		var2 (string): the variable that should have inconsistent values removed
		constraint (BinaryConstraint): the constraint connecting var1 and var2
	Returns:
		set<tuple<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""
def revise(assignment, csp, var1, var2, constraint):
	inferences = set([])
	domains = assignment.varDomains

	constraints = csp.binaryConstraints
	for value in assignment.varDomains[var2]:
		canBeSatisifed = False
		#getting the value of the other variable
		for otherValue in domains[var1]:
			status = constraint.isSatisfied(otherValue, value)
			if status:
				canBeSatisifed = True
		if not canBeSatisifed:
			newInference = (var2, value)
			inferences.add(newInference)


	incomplete = False
	for update in inferences:
		var1 = update[0]
		value1 = update[1]
		domains[var1].remove(value1)
		if len(domains[var1]) == 0:
			incomplete = True

	if incomplete:
		for update in inferences:
			var1 = update[0]
			value1 = update[1]
			domains[var1].add(value1)
		return None

	return inferences


"""
	Implements the maintaining arc consistency algorithm.
	Inferences take the form of (variable, value) where the value is being removed from the
	domain of variable. This format is important so that the inferences can be reversed if they
	result in a conflicting partial assignment. If the algorithm reveals an inconsistency, and
	inferences made should be reversed before ending the fuction.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
		var (string): the variable that has just been assigned a value
		value (string): the value that has just been assigned
	Returns:
		set<<variable, value>>
		the inferences made in this call or None if inconsistent assignment
"""
def maintainArcConsistency(assignment, csp, var, value):
	inferences = set([])
	
	constraints = csp.binaryConstraints
	domains = assignment.varDomains
	queue = Queue()
	visited = []

	#finding neighbors
	neighbors = findNeighbors(var, constraints)

	#adds neighbors to the queue 
	for neighbor in neighbors:
		queue.enqueue(neighbor)


	while not queue.isEmpty():
		constraint = queue.deque()

		#keep track of the nodes visited
		visited.append(constraint[0])

		#find the inference
		inference = revise(assignment, csp, constraint[0], constraint[1], constraint[2])

		#if inference is None, an inconsistency has been found.  Undo all assignemnts and return None in turn
		if inference == None:
			for update in inferences:
				var1 = update[0]
				value1 = update[1]
				domains[var1].add(value1)
			return None

		#add the inference, and add the neighbors associated with the inference to the queue	
		elif inference != set([]):
			inference = (list(inference))
			inference = tuple(inference[0])
			inferences.add(inference)
			neighbors = findNeighbors(constraint[1], constraints)
			for neighbor in neighbors:
				queue.enqueue(neighbor)

	return inferences

"""helper method to find neighbors"""
def findNeighbors(var, constraints):
	neighbors = []

	for item in constraints:
		if item.var1 == var:
			neighbors.append((var, item.var2, item))
		elif item.var2 == var:
			neighbors.append((var, item.var1, item))

	return neighbors

"""
	AC3 algorithm for constraint propogation. Used as a preprocessing step to reduce the problem
	before running recursive backtracking.

	Args:
		assignment (Assignment): the partial assignment to expand
		csp (ConstraintSatisfactionProblem): the problem description
	Returns:
		Assignment
		the updated assignment after inferences are made or None if an inconsistent assignment
"""
def AC3(assignment, csp):
	inferences = set([])
	
	constraints = csp.binaryConstraints
	domains = assignment.varDomains
	queue = Queue()
	visited = []

	#add all variables associated with the constraints
	for constraint in constraints:
		queue.enqueue((constraint.var1, constraint.var2, constraint))
		queue.enqueue((constraint.var2, constraint.var1, constraint))

	#repeat the same process as above from maintainArcConsistency	
	while not queue.isEmpty():
		constraint = queue.deque()

		visited.append(constraint[0])
		inference = revise(assignment, csp, constraint[0], constraint[1], constraint[2])
		if inference == None:
			for update in inferences:
				var1 = update[0]
				value1 = update[1]
				domains[var1].add(value1)
			return None

		elif inference != set([]):
			inference = (list(inference))
			inference = tuple(inference[0])
			inferences.add(inference)
			neighbors = findNeighbors(constraint[1], constraints)
			for neighbor in neighbors:
				queue.enqueue(neighbor)

	return assignment


"""
	Solves a binary constraint satisfaction problem.

	Args:
		csp (ConstraintSatisfactionProblem): a CSP to be solved
		orderValuesMethod (function): a function to decide the next value to try
		selectVariableMethod (function): a function to decide which variable to assign next
		inferenceMethod (function): a function to specify what type of inferences to use
		useAC3 (boolean): specifies whether to use the AC3 preprocessing step or not
	Returns:
		dictionary<string, value>
		A map from variables to their assigned values. None if no solution exists.
"""
def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic, selectVariableMethod=minimumRemainingValuesHeuristic, inferenceMethod=forwardChecking, useAC3=True):
	assignment = Assignment(csp)

	assignment = eliminateUnaryConstraints(assignment, csp)
	if assignment == None:
		return assignment

	if useAC3:
		assignment = AC3(assignment, csp)
		if assignment == None:
			return assignment

	assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
	if assignment == None:
		return assignment

	return assignment.extractSolution()