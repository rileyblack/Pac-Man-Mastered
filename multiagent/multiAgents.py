# Search based-agents implemented by Riley Black
#
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        Chooses among the best options according to the evaluation function by
        taking a GameState and returning some Directions.X for some X in the
        set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function uses the current and proposed successor game states
        to evaluate the move and returns a quantifying rating, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        newFoodList = newFood.asList()
        currentFoodList = currentFood.asList()
        currentCapsules = currentGameState.getCapsules()
        newCapsules = successorGameState.getCapsules()

        if successorGameState.isWin():
            return 99999

        if successorGameState.isLose():
            return -99999

        score = successorGameState.getScore()

        # Rewarding/penalizing score if scared/not scared ghosts are close
        ghostIndex = 0
        for ghostPosition in newGhostPositions:
            distance = util.manhattanDistance(ghostPosition, newPos)
            howCloseIsGhost = max(0, 5 - distance)
            ghostScareTime = newScaredTimes[ghostIndex]
            if ghostScareTime > 2:
                score = score + (3 * howCloseIsGhost)
            else:
                score = score - (3 * howCloseIsGhost)
            ghostIndex = ghostIndex + 1

        # Rewarding score if food is being eaten and penalizing score if food is far
        nearestFood = 99999
        for food in newFoodList:
            distance = util.manhattanDistance(food, newPos)
            if distance < nearestFood:
                nearestFood = distance

        score = score - nearestFood

        if len(currentFoodList) > len(newFoodList):
            score = score + 100  # if food was eaten, reward score by 100 points

        # Rewarding score if power capsule is being eaten and penalizing score if power capsule is far
        if len(newCapsules) != 0:
            nearestCapsule = 99999
            for capsulePosition in newCapsules:
                distance = util.manhattanDistance(capsulePosition, newPos)
                if (distance < nearestCapsule):
                    nearestCapsule = distance

            score = score - nearestCapsule / 2

            if len(currentCapsules) > len(newCapsules):
                score = score + 200  # if capsule was eaten, reward score by 200 points

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Minimax searcher
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """

        def maxValue(state, depth, index, ghostCount):
            """
            Returns the optimal agent move for a given scenario assuming the ghosts always choose worst case.

             state : a GameState object (pacman.py) of the current scenario
             depth : an integer of the remaining minimax iterations
             index : an integer identifier of the agent
             ghostCount : an integer of the number of ghosts in the current scenario

             Return : tuple containing the best move and associated score
            """
            if state.isWin() or state.isLose() or depth == 0:
                bestMove = None
                bestMoveScore = self.evaluationFunction(state)
                return bestMove, bestMoveScore

            bestMove = Directions.STOP
            bestMoveScore = -99999

            viableMoves = state.getLegalActions(index)

            for currentMove in viableMoves:
                successor = state.generateSuccessor(index, currentMove)
                move, currentMoveScore = minValue(successor, depth, index + 1, ghostCount)

                if currentMoveScore > bestMoveScore:
                    bestMove = currentMove
                    bestMoveScore = currentMoveScore

            return bestMove, bestMoveScore

        def minValue(state, depth, index, ghostCount):
            """
            Returns the optimal agent move for a given scenario assuming the ghosts always choose worst case.

             state : a GameState object (pacman.py) of the current scenario
             depth : an integer of the remaining minimax iterations
             index : an integer identifier of the agent
             ghostCount : an integer of the number of ghosts in the current scenario

             Return : tuple containing the best move and associated score
            """
            if state.isWin() or state.isLose() or depth == 0:
                worstMove = None
                worstMoveScore = self.evaluationFunction(state)
                return worstMove, worstMoveScore

            worstMove = Directions.STOP
            worstMoveScore = 99999

            viableMoves = state.getLegalActions(index)

            for currentMove in viableMoves:
                successor = state.generateSuccessor(index, currentMove)
                if index < ghostCount:
                    move, currentMoveScore = minValue(successor, depth, index + 1, ghostCount)
                else:
                    move, currentMoveScore = maxValue(successor, depth - 1, 0, ghostCount)

                if currentMoveScore < worstMoveScore:
                    worstMove = currentMove
                    worstMoveScore = currentMoveScore

            return worstMove, worstMoveScore

        depth = self.depth  # retrieve depth of minimax
        startIndex = 0  # begin agent index with Pacman aka 0
        ghostNumber = gameState.getNumAgents() - 1  # retrieve number of ghosts (minus 1 for Pacman agent)
        move, moveScore = maxValue(gameState, depth, startIndex, ghostNumber)  # find Pacmans best move
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Expectimax searcher
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        def maxValue(state, depth, index, ghostCount):
            """
            Returns the optimal agent move for a given scenario assuming the ghosts always choose randomly.

             state : a GameState object (pacman.py) of the current scenario
             depth : an integer of the remaining minimax iterations
             index : an integer identifier of the agent
             ghostCount : an integer of the number of ghosts in the current scenario

             Return : tuple containing the best move and associated score
            """
            if state.isWin() or state.isLose() or depth == 0:
                bestMove = None
                bestMoveScore = self.evaluationFunction(state)
                return bestMove, bestMoveScore

            bestMove = Directions.STOP
            bestMoveScore = -99999

            viableMoves = state.getLegalActions(index)

            for currentMove in viableMoves:
                successor = state.generateSuccessor(index, currentMove)
                move, currentMoveScore = expValue(successor, depth, index + 1, ghostCount)

                if currentMoveScore > bestMoveScore:
                    bestMove = currentMove
                    bestMoveScore = currentMoveScore

            return bestMove, bestMoveScore

        def expValue(state, depth, index, ghostCount):
            """
            Returns the expected agent move for a given scenario assuming the ghosts always choose randomly.

             state : a GameState object (pacman.py) of the current scenario
             depth : an integer of the remaining minimax iterations
             index : an integer identifier of the agent
             ghostCount : an integer of the number of ghosts in the current scenario

             Return : tuple containing the expected move and associated score
            """
            if state.isWin() or state.isLose() or depth == 0:
                bestMove = None
                bestMoveScore = self.evaluationFunction(state)
                return bestMove, bestMoveScore

            move = Directions.STOP
            moveScore = 0

            viableMoves = state.getLegalActions(index)
            viableMovesCount = len(viableMoves)

            for currentMove in viableMoves:
                successor = state.generateSuccessor(index, currentMove)
                if index < ghostCount:
                    move, currentMoveScore = expValue(successor, depth, index + 1, ghostCount)
                else:
                    move, currentMoveScore = maxValue(successor, depth - 1, 0, ghostCount)

                moveScore = moveScore + currentMoveScore

            expectedScore = moveScore/viableMovesCount
            return move, expectedScore

        depth = self.depth  # retrieve depth of expectimax
        startIndex = 0  # begin agent index with Pacman aka 0
        ghostNumber = gameState.getNumAgents() - 1  # retrieve number of ghosts (minus 1 for Pacman agent)
        move, moveScore = maxValue(gameState, depth, startIndex, ghostNumber)  # call max value since Pacman begins
        return move
