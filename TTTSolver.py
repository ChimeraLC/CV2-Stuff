"""
Relatively basic move recommendor for tic tac toe
"""
def Solve(grid):
	threes = [[0,1,2], [3,4,5], [6,7,8], 
	   [0,3,6], [1,4,7], [2,5,8], 
	   [0,4,8], [2, 4, 6]]
	# Figure out whose move it is
	difcount = 0
	player = 0
	for val in grid:
		if (val == 0):
			difcount -=1
		if (val == 1):
			difcount += 1
	# x turn
	if difcount == 0:
		player = 1
	# y turn
	elif difcount == 1:
		player = 0
	else:
		return -1, 0
	
	# calculating all the positions
	"""
	for index in len(threes):
	for i in range(3):
		threes[index[i]] = grid[threes[index[i]]]
	"""
	# looking for wins	
	for t in threes:
		if (sorted([grid[i] for i in t]) == [player, player, 2]):
			for i in t:
				if grid[i] == 2:
					return i, player
	# preventing enemy wins
	for t in threes:
		if (sorted([grid[i] for i in t]) == [1-player, 1-player, 2]):
			for i in t:
				if grid[i] == 2:
					return i, player
	# finding forks
	for pos in range(9):
		# Test each possible move for forks
		if grid[pos] != 2:
			continue
		gridcopy = grid.copy()
		gridcopy[pos] = player
		if FindMatches(gridcopy, [player, player, 2]) >= 2:
			return pos, player
	# denying forks
	for pos in range(9):
		# Test each possible move for forks
		if grid[pos] != 2:
			continue
		gridcopy = grid.copy()
		gridcopy[pos] = player
		if FindMatches(gridcopy, [1-player, 1-player, 2]) >= 2:
			return pos, player
	# choosing closest to center
	if grid[4] == 2:
		return 4, player
	# choosing random
	for pos in range(9):
		if grid[pos] == 2:
			return pos, player
	return -1, player
	

# Defines how good a position is for a player
# 3- won 2 - preventing loss 3 - forced win 4 - setup win 5 - aim towards center
def Score(grid, player):
	# Winning states
    if FindMatches(grid, [player, player, player]) >= 1:
	    return 3

# Finds opportunites for wins
def FindMatches(grid, distribution):
	ret = 0
	# rows 
	for i in range(3):
		if sorted(grid[i*3:i*3+3]) == distribution:
			ret += 1
	# columns
	for j in range(3):
		if sorted([grid[j], grid[3+j], grid[6+j]]) == distribution:
			ret += 1
	# diagonals
	if sorted([grid[0], grid[4], grid[8]]) == distribution:
		ret += 1
	if sorted([grid[2], grid[4], grid[6]]) == distribution:
		ret += 1
	return ret

threes = [[0,1,2], [3,4,5], [6,7,8], 
	   [0,3,6], [1,4,7], [2,5,8], 
	   [0,4,8], [2, 4, 6]]
grid = [2, 2, 2, 1, 2, 1, 0, 0, 2]
