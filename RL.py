# ----------------------------- IMPORTS & STUFF -------------------------------------------
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

# ----------------------------- FUNCTIONS -------------------------------------------
def drawTrack(States_history,loop):
    
    plt.figure()

    # Add agent
    xValues = []
    yValues = []
    for State in States_history:
        xValues.append(State[0])
        yValues.append(State[1])
    plt.plot(xValues,yValues,'ro',linestyle="--")
    plt.axis([-1, gridSizeX+1, -1, gridSizeY+1])
    plt.grid('on', linestyle='--')

    # Add track limits
    outerLimit = plt.Circle((0,0),outerTrackLimits,color = 'k', fill =False)
    innerLimit = plt.Circle((0,0),innerTrackLimits,color = 'k', fill =False)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(outerLimit)
    ax.add_patch(innerLimit)

    # Add plot info
    plt.title('Loop number ' + str(loop))
    plt.xlabel('x-position')
    plt.ylabel('y-position')


def createSpaces():
    stateSpace = [] 
    for x_cor in range(0,gridSizeX+1):
        for y_cor in range(0,gridSizeY+1):
            for speed in range(topSpeed+1):
                for direction in np.linspace(0,360-maxSteering,int(360/maxSteering)):
                    stateSpace.append(np.array([x_cor,y_cor,speed,direction],dtype = int))
             
    actionSpace = []
    for acceleration in [maxBrake,0,maxThrottle]:
        for steering in [maxSteering,0,-maxSteering]:
            actionSpace.append(np.array([acceleration,steering],dtype = int))

    return stateSpace,actionSpace


def convert2Index(stateOrAction):

    # In case the index is wanted for an action
    if len(stateOrAction) == 2:
        counter = 0
        for checkedAction in actionSpace:
            if stateOrAction[0] == checkedAction[0] and stateOrAction[1] == checkedAction[1]:
                stateActionIndex = counter
                break
            counter += 1        

    # In case the index is wanted for a state
    elif len(stateOrAction) == 4:
        stateActionIndex = int(stateOrAction[0]*xCoordValue + stateOrAction[1]*yCoordValue + stateOrAction[2]*speedValue + stateOrAction[3]*directionValue)

    return stateActionIndex


def createMatrices():
    P = np.zeros((len(stateSpace),len(actionSpace)), dtype = int) # Contains immediate reward
    M = np.zeros((len(stateSpace),len(actionSpace)), dtype = int) # Contains index for next state
    Q = np.zeros((len(stateSpace),len(actionSpace))) # Contains Q-values

    for row in range(len(P)):
        for col in range(len(P[0])):
            State = stateSpace[row]
            Action = actionSpace[col]

            nextState = movement(State,Action)
            [finished,crashed] = checkeredFlag(nextState)
            P[row][col] = pointScoring(nextState,finished,crashed)

            nextIndex = convert2Index(nextState)
            M[row][col] = nextIndex
    

    return P,Q,M


def performAction(State,Q):
    # Make the decision on what to do based on position and velocity
    
    # State - Consist of current state [x-cor,y-cor,speed,direction] 
    # Action with the index of the highest Q-value in the row corresponding to current state
    # If multiple actions have the same Q-value, a random one is chosen     
    Qrow = (Q[convert2Index(State)])
    maxValue = crashedScore
    equalValuedIndex = []
    for col in range(len(Qrow)):
        if Qrow[col] > maxValue:
            maxValue = Qrow[col]
            equalValuedIndex = [col]
        elif Qrow[col] == maxValue:
            equalValuedIndex.append(col)

    bestAction = actionSpace[random.choice(equalValuedIndex)] # Only randomized between actions of equal Q-values
    
    # Make selection from best and random actions, based on weight
    Action = random.choices(np.concatenate((actionSpace,[bestAction]),axis=0),cum_weights = actionWeight,k=1)[0]

    return Action


def checkeredFlag(State):
    if State[1] <= 0: #and State[0] > innerTrackLimits:
        finished = True
    else:
        finished = False

    positionRadius = np.sqrt(State[0]**2 + State[1]**2)
    if positionRadius > outerTrackLimits or positionRadius < innerTrackLimits:
        crashed = True
    else:
        crashed = False
    
    return finished,crashed


def movement(State,Action):
    # Gets latest position, velocity and Action to calculate new values for position and velocity    
    xDelta = int(State[2]*math.cos(State[3]*math.pi/180))
    yDelta = int(State[2]*math.sin(State[3]*math.pi/180))
    
    delta_State = np.array([xDelta,yDelta,Action[0],Action[1]])
    new_State = State + delta_State

    # Limit max speed and min speed (limiting number of states)
    if new_State[2] >= topSpeed:
        new_State[2] = topSpeed
    elif new_State[2] <= 0:
        new_State[2] = 0

    # Limit movement to be inside State space
    if new_State[0] < 0:
        new_State[0] = 0
    elif new_State[0] > gridSizeX:
        new_State[0] = gridSizeX

    if new_State[1] < 0:
        new_State[1] = 0
    elif new_State[1] > gridSizeY:
        new_State[1] = gridSizeY

    # No turning possible if not moving (OPTIONAL: Will represent a car rather than track vehicle)
    if State[2] == 0:
        new_State[3] = State[3]

    # Always display direction as between 0 and 359 deg
    if new_State[3] < 0:
        new_State[3] += 360
    elif new_State[3] >= 360:
        new_State[3] -= 360

    return new_State


def pointScoring(State,finished,crashed):
    # Default for taking a time step
    pointScored = timeScore

    # Score for finishing
    if finished:
        pointScored = finishScore
    
    elif crashed:
        pointScored = crashedScore

    return pointScored


def QLearning(Q,State,Action,nextState,reward):
    row = convert2Index(State)
    col = convert2Index(Action)

    maxQ = max(Q[convert2Index(nextState)])
    Q[row][col] = (1-alpha)*Q[row][col] + alpha*(reward + gamma*maxQ)

    return Q


# ----------------------------- MAIN ------------------------------------------
# Define track parameters
gridSizeX = 20
gridSizeY = 20
outerTrackLimits = gridSizeX
innerTrackLimits = gridSizeX*(3/4)
start_position = [0,(innerTrackLimits+(outerTrackLimits-innerTrackLimits)/2)]
start_velocity = [0,0]

# Define agent parameters
maxSteering = 45 # [Deg]
maxThrottle = 1
maxBrake = -1
topSpeed = 3

# How much each parameter affect the indexing of a state:
# x-coordinate only changes when all y-coordinate-, speed- and direction combinations have been run through 
directionValue = 1/maxSteering
speedValue = 360/maxSteering
yCoordValue = (topSpeed+1)*speedValue # Include speed = 0
xCoordValue = (gridSizeY+1)*yCoordValue # Include y = 0

# Define scoring values
timeScore = -1
crashedScore = -20
finishScore = 40

# Make different policies
alpha = 0.3 # alpha -> 0: No updated Q, alpha -> 1: Fully updated Q
gamma = 0.7 # gamma -> 0: Greedy, short-term focused, gamma -> 1: Long-term focused 
epsilon_original = 0.5 # epsilon -> 0: More exploration, epsilon -> 1: Take known route
epsilon = epsiolon_original

# State- and action spaces
# List of lists of all possible states and action: [x-coordinates,y-coordinates,Velocities,Directions]
[stateSpace,actionSpace] = createSpaces()

# Create reward- (P), learning- (Q) and movement- (M) matrices    
[P,Q,M] = createMatrices()

# Weights for randomizing action taking
actionWeight = np.concatenate(([(1-epsilon)/len(actionSpace)*k for k in range(1,len(actionSpace)+1)],[1]))

# Learning loop
timesFinishedTraining = 0
timesFinishedDemo = 0
diffArr = []
learningLoops = 50000
demoStart = int(learningLoops*9/10)
loops2Draw = [1,int(learningLoops/2),learningLoops-5,learningLoops-4,learningLoops-3,learningLoops-2,learningLoops-1,learningLoops]
for loopNumber in range(1,learningLoops+1):
    oldQ = copy.deepcopy(Q)
    
    # Change behaviour when the groundwork has been done
    if loopNumber == demoStart:
        epsilon = 1

    # Create framework for lists of history
    States_history = [np.array(start_position + start_velocity,dtype = int)] # List of state parameters [x-coordinate, y-coordinate, Speed, Angle (Deg)]
    Actions_history = [np.array([0,0],dtype = int)] # List of velocity changes, elements consisting of [Acceleration in velocity diretion, Steering in deg from vel-vector]
    finished = False
    crashed = False    
    # One drive loop
    while not finished and not crashed:
        # Make decision on how to drive and add to history
        Action = performAction(States_history[-1],Q)
        Actions_history.append(Action)

        # Calculate next state and add to history
        new_State = stateSpace[M[convert2Index(States_history[-1])][convert2Index(Action)]]
        States_history.append(new_State)

        # Check if lap is finished
        [finished,crashed] = checkeredFlag(new_State)
        
        if finished:
            if loopNumber < demoStart:
                timesFinishedTraining += 1
            else:
                timesFinishedDemo += 1

        # Get rewarded
        pointScored = pointScoring(new_State,finished,crashed)

        # Learning 
        # States_history[-2] and [-1] are the original and new state in this time step
        Q = QLearning(Q,States_history[-2],Action,States_history[-1],pointScored) 

    # Compare change in Q 
    diffTot = np.sum(abs(Q-oldQ))
    a = np.sum(oldQ)
    if a == 0:
        diffPercentage = 0
    else:
        diffPercentage = abs(diffTot/a)
    diffArr.append(diffPercentage)

    # Draw this learning loop
    if loopNumber in loops2Draw:
        drawTrack(States_history,loopNumber)

print('Number of possible states: ' + str(len(stateSpace)),'\nParameters are:\nAlpha: ' 
+ str(alpha) + '\nGamma: ' + str(gamma) + '\nEpsilon: ' + str(epsilon_original) + 
'\nNumber of learning loops: ' + str(learningLoops))
print('Finishes during training: ' + str(timesFinishedTraining) + ', \nequal to ' + str(round(100*timesFinishedTraining/demoStart,2))
+ '% of the runs \nFinishes during demonstration: ' +  str(timesFinishedDemo) + ', \nequal to '
+ str(round(100*timesFinishedDemo/(learningLoops-demoStart),2)) + ' % of the runs')


# Plot error in Q-matrices
plt.figure()
plt.plot(list(range(1,learningLoops+1)),diffArr)
plt.yscale("log")
plt.title('Error between Q-matrices of this and previous loop')
plt.xlabel('Loop number')
plt.ylabel('Error [%]')


# Display plot
plt.show()








