# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import util
import random
import busters
import game


class InferenceModule:
	"""
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

	# ###########################################
	# Useful methods for all inference modules #
	############################################

	def __init__(self, ghostAgent):
		"Sets the ghost agent for later access"
		self.ghostAgent = ghostAgent
		self.index = ghostAgent.index
		self.obs = []  # most recent observation position

	def getJailPosition(self):
		return (2 * self.ghostAgent.index - 1, 1)

	def getPositionDistribution(self, gameState):
		"""
        Returns a distribution over successor positions of the ghost from the given gameState.

        You must first place the ghost in the gameState, using setGhostPosition below.
        """
		ghostPosition = gameState.getGhostPosition(self.index)  # The position you set
		actionDist = self.ghostAgent.getDistribution(gameState)
		dist = util.Counter()
		for action, prob in actionDist.items():
			successorPosition = game.Actions.getSuccessor(ghostPosition, action)
			dist[successorPosition] = prob
		return dist

	def setGhostPosition(self, gameState, ghostPosition):
		"""
        Sets the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of the
        GameState object which is responsible for maintaining game state, not a
        reference to the original object.  Note also that the ghost distance
        observations are stored at the time the GameState object is created, so
        changing the position of the ghost will not affect the functioning of
        observeState.
        """
		conf = game.Configuration(ghostPosition, game.Directions.STOP)
		gameState.data.agentStates[self.index] = game.AgentState(conf, False)
		return gameState

	def observeState(self, gameState):
		"Collects the relevant noisy distance observation and pass it along."
		distances = gameState.getNoisyGhostDistances()
		if len(distances) >= self.index:  # Check for missing observations
			obs = distances[self.index - 1]
			self.obs = obs
			self.observe(obs, gameState)

	def initialize(self, gameState):
		"Initializes beliefs to a uniform distribution over all positions."
		# The legal positions do not include the ghost prison cells in the bottom left.
		self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
		self.initializeUniformly(gameState)

	######################################
	# Methods that need to be overridden #
	######################################

	def initializeUniformly(self, gameState):
		"Sets the belief state to a uniform prior belief over all positions."
		pass

	def observe(self, observation, gameState):
		"Updates beliefs based on the given distance observation and gameState."
		pass

	def elapseTime(self, gameState):
		"Updates beliefs for a time step elapsing from a gameState."
		pass

	def getBeliefDistribution(self):
		"""
        Returns the agent's current belief state, a distribution over
        ghost locations conditioned on all evidence so far.
        """
		pass


class ExactInference(InferenceModule):
	"""
    The exact dynamic inference module should use forward-algorithm
    updates to compute the exact belief function at each time step.
    """

	def initializeUniformly(self, gameState):
		"Begin with a uniform distribution over ghost positions."
		self.beliefs = util.Counter()
		for p in self.legalPositions: self.beliefs[p] = 1.0
		self.beliefs.normalize()

	def observe(self, observation, gameState):
		"""
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

        The emissionModel below stores the probability of the noisyDistance for any true
        distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

        self.legalPositions is a list of the possible ghost positions (you
        should only consider positions that are in self.legalPositions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated so
             that the ghost appears in its prison cell, position self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).

        """
		noisyDistance = observation
		emissionModel = busters.getObservationDistribution(noisyDistance)
		pacmanPosition = gameState.getPacmanPosition()

		#If the noisy distance is none, then a ghost is in jail
		if noisyDistance == None:
			pos = self.getJailPosition()
			#set the belief state at the jail position to be 1
			self.beliefs[pos] = 1
			#set all the other positions to be zero
			for position in self.legalPositions:
				if position != pos:
					self.beliefs[position] = 0
			return


		# Go through each legal position. Apply Baye's Rule.
		for position in self.legalPositions:
			true_distance = util.manhattanDistance(pacmanPosition, position)
			self.beliefs[position] = emissionModel[true_distance] * self.beliefs[position]

		
		allPossible = self.beliefs
		allPossible.normalize()
		self.beliefs = allPossible

	def elapseTime(self, gameState):
		"""
        Update self.beliefs in response to a time step passing from the current state.

        The transition model is not entirely stationary: it may depend on Pacman's
        current position (e.g., for DirectionalGhost).  However, this is not a problem,
        as Pacman's current position is known.

        In order to obtain the distribution over new positions for the
        ghost, given its previous position (oldPos) as well as Pacman's
        current position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        Note that you may need to replace "oldPos" with the correct name
        of the variable that you have used to refer to the previous ghost
        position for which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in self.legalPositions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
        in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern
        yourself), the line of code at the top of this comment block for obtaining newPosDist makes
        use of two helper methods provided in InferenceModule above:

          1) self.setGhostPosition(gameState, ghostPosition)
              This method alters the gameState by placing the ghost we're tracking
              in a particular position.  This altered gameState can be used to query
              what the ghost would do in this position.

          2) self.getPositionDistribution(gameState)
              This method uses the ghost agent to determine what positions the ghost
              will move to from the provided gameState.  The ghost must be placed
              in the gameState with a call to self.setGhostPosition above.

        It is worthwhile, however, to understand why these two helper methods are used and how they
        combine to give us a belief distribution over new positions after a time update from a particular position
        """

		# go through each legal position and sum all the probabilities of a ghost moving to that position
		update = util.Counter()
		for oldPos in self.legalPositions:
			newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
			for newPos, prob in newPosDist.items():
				update[newPos] = update[newPos] + (prob * self.beliefs[oldPos])

		self.beliefs = util.normalize(update)


	def getBeliefDistribution(self):
		return self.beliefs


class ParticleFilter(InferenceModule):
	"""
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses
    an element from a list uniformly at random, and util.sample, which
    samples a key from a Counter by treating its values as probabilities.
    """


	def __init__(self, ghostAgent, numParticles=300):
		InferenceModule.__init__(self, ghostAgent);
		self.setNumParticles(numParticles)

	def setNumParticles(self, numParticles):
		self.numParticles = numParticles


	def initializeUniformly(self, gameState):
		"""
          Initializes a list of particles. Use self.numParticles for the number of particles.
          Use self.legalPositions for the legal board positions where a particle could be located.
          Particles should be evenly (not randomly) distributed across positions in order to
          ensure a uniform prior.

          ** NOTE **
            the variable you store your particles in must be a list; a list is simply a collection
            of unweighted variables (positions in this case). Storing your particles as a Counter or
            dictionary (where there could be an associated weight with each position) is incorrect
            and will produce errors
        """
		
		#Evenly distribute the ghost positions across the particles
		self.particleList = []
		for position in self.legalPositions:
			for i in range(self.numParticles / len(self.legalPositions)):
				self.particleList.append(position)

	def observe(self, observation, gameState):
		"""
        Update beliefs based on the given distance observation. Make
        sure to handle the special case where all particles have weight
        0 after reweighting based on observation. If this happens,
        resample particles uniformly at random from the set of legal
        positions (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, **all** particles should be updated so
             that the ghost appears in its prison cell, self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).

          2) When all particles receive 0 weight, they should be recreated from the
             prior distribution by calling initializeUniformly. The total weight
             for a belief distribution can be found by calling totalCount on
             a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution

        You may also want to use util.manhattanDistance to calculate the distance
        between a particle and pacman's position.
        """

		noisyDistance = observation
		emissionModel = busters.getObservationDistribution(noisyDistance)
		pacmanPosition = gameState.getPacmanPosition()
		

		#If the noisy distance is none, a ghost is in jail
		if noisyDistance == None:
			pos = self.getJailPosition()
			self.particleList = []
			#place all the particles in the jail position to reflect our belief
			for i in range(self.numParticles):
				self.particleList.append(pos)
			#set the probability of being in a non-jail cell to be zero.  Otherwise, the jail cell prob is one
			for position in self.legalPositions:
				if position != pos:
					self.beliefDistribution[position] = 0
				else:
					self.beliefDistribution[position] = 1

			return self.beliefDistribution


		beliefs = self.beliefDistribution
		newBeliefs = util.Counter()

		#go through each particel and update it's probability based on Baye's ruel
		for position in self.particleList:
			true_distance = util.manhattanDistance(pacmanPosition, position)
			newBeliefs[position] += emissionModel[true_distance]

		#normalize
		newBeliefs.normalize()
		self.beliefDistribution = newBeliefs

		#resample the particles to reflect the change in the belief state
		self.particleList = []
		if (newBeliefs.totalCount() == 0):
			#if the total count is zero, reinitialize
			self.initializeUniformly(gameState)
		else:
			# sample the number of particles to get our new particle positions
			for pos in range(self.numParticles):
				self.particleList.append(util.sample(newBeliefs))

		return newBeliefs

	def elapseTime(self, gameState):
		"""
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given
        its previous position (oldPos) as well as Pacman's current
        position.

        util.sample(Counter object) is a helper method to generate a sample from a
        belief distribution
        """
		
		#go through each particle and calculate the probability of going to another legal position
		#sum these up and you're good to go
		update = []
		for oldPos in self.particleList:
			newDistr = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
			update.append(util.sample(newDistr))
		self.particleList = update



	def getBeliefDistribution(self):
		"""
          Return the agent's current belief state, a distribution over
          ghost locations conditioned on all evidence and time passage. This method
          essentially converts a list of particles into a belief distribution (a Counter object)
        """
	
		#simply adds up the number of times each position appears and normalizes.  Give the belief distribution.  
		beliefDistr = util.Counter()
		for pos in self.particleList:
			beliefDistr[pos] += 1
		
		beliefDistr.normalize()
		self.beliefDistribution = beliefDistr
		return beliefDistr


class MarginalInference(InferenceModule):
	"A wrapper around the JointInference module that returns marginal beliefs about ghosts."

	def initializeUniformly(self, gameState):
		"Set the belief state to an initial, prior value."
		if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
		jointInference.addGhostAgent(self.ghostAgent)

	def observeState(self, gameState):
		"Update beliefs based on the given distance observation and gameState."
		if self.index == 1: jointInference.observeState(gameState)

	def elapseTime(self, gameState):
		"Update beliefs for a time step elapsing from a gameState."
		if self.index == 1: jointInference.elapseTime(gameState)

	def getBeliefDistribution(self):
		"Returns the marginal belief over a particular ghost by summing out the others."
		jointDistribution = jointInference.getBeliefDistribution()
		dist = util.Counter()
		for t, prob in jointDistribution.items():
			dist[t[self.index - 1]] += prob
		return dist