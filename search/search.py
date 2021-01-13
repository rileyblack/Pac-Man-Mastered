# Search algorithms implemented by Riley Black
#
# search.py
# ---------
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

import util

class SearchProblem:
    """
    Abstract search problem class structure.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
         state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
         state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Searches the deepest nodes in the search tree first.
    """
    current_path = []
    visited_positions = []
    fringe = util.PriorityQueue()

    fringe.push((problem.getStartState(), current_path), 0)

    while not fringe.isEmpty():
        current_position, current_path = fringe.pop()

        if problem.isGoalState(current_position):
            return current_path

        if current_position not in visited_positions:
            visited_positions.append(current_position)
            successors = problem.getSuccessors(current_position)
            if successors:
                successors.reverse()

            for successor in successors:
                successor_position = successor[0]
                if successor_position not in visited_positions:
                    successor_direction = successor[1]
                    successor_path = current_path + [successor_direction]
                    successor_depth = len(successor_path)
                    fringe.push((successor_position, successor_path), -successor_depth)
                    # negative successor_depth as priority so priority queue pops "deeper" successors sooner
    return []

def breadthFirstSearch(problem):
    """
    Searches the shallowest nodes in the search tree first.
    """
    current_path = []
    visited_positions = []
    fringe = util.Queue()  # priority queue not needed since successors are pushed/popped in bfs order naturally

    fringe.push((problem.getStartState(), current_path))

    while not fringe.isEmpty():
        current_position, current_path = fringe.pop()

        if problem.isGoalState(current_position):
            return current_path

        if current_position not in visited_positions:
            visited_positions.append(current_position)
            successors = problem.getSuccessors(current_position)

            for successor in successors:
                successor_position = successor[0]
                if successor_position not in visited_positions:
                    successor_direction = successor[1]
                    successor_path = current_path + [successor_direction]
                    fringe.push((successor_position, successor_path))
    return []

def uniformCostSearch(problem):
    """
    Searches the node of least total cost first.
    """
    current_path = []
    visited_positions = []
    fringe = util.PriorityQueue()

    fringe.push((problem.getStartState(), current_path), 0)

    while not fringe.isEmpty():
        current_position, current_path = fringe.pop()

        if problem.isGoalState(current_position):
            return current_path

        if current_position not in visited_positions:
            visited_positions.append(current_position)
            successors = problem.getSuccessors(current_position)

            for successor in successors:
                successor_position = successor[0]
                if successor_position not in visited_positions:
                    successor_direction = successor[1]
                    successor_path = current_path + [successor_direction]
                    successor_cost = problem.getCostOfActions(successor_path)
                    fringe.push((successor_position, successor_path), successor_cost)
    return []

def nullHeuristic(state, problem=None):
    """
    Trivial heuristic.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    current_path = []
    visited_positions = []
    fringe = util.PriorityQueue()

    fringe.push((problem.getStartState(), current_path), 0)

    while not fringe.isEmpty():
        current_position, current_path = fringe.pop()

        if problem.isGoalState(current_position):
            return current_path

        if current_position not in visited_positions:
            visited_positions.append(current_position)
            successors = problem.getSuccessors(current_position)

            for successor in successors:
                successor_position = successor[0]
                if successor_position not in visited_positions:
                    successor_direction = successor[1]
                    successor_path = current_path + [successor_direction]
                    successor_cost = problem.getCostOfActions(successor_path) + heuristic(successor_position, problem)
                    fringe.push((successor_position, successor_path), successor_cost)
    return []


# Abbreviations
dfs = depthFirstSearch
bfs = breadthFirstSearch
ucs = uniformCostSearch
astar = aStarSearch
