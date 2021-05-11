import json, ast
import numpy as np
import random
from capture import runGames
from capture import readCommand


GAMMA = 0.95
ALPHA = 0.05

def readRecords(filename):
	records = []
	with open(filename, "r") as f:
		for line in f.readlines():
			record = ast.literal_eval(line)
			records.append(record)
	return records


def train(epoch, records, weights):
	ret = [0] * len(weights)
	for e in range(epoch):
		# shuffle records
		random.shuffle(records)
		w = weights[:]
		for r in records:
			p = ast.literal_eval(r["pre"])
			c = ast.literal_eval(r["cur"])
			prev_state = [float(x) for x in p]
			curr_state = [float(x) for x in c]
			reward = float(r["r"])
			prev_q = qFunc(prev_state, w)
			curr_q = qFunc(curr_state, w)
			# i = random.choice(range(len(w)))
			# delta = ALPHA * (reward + GAMMA * curr_q - prev_q) * prev_state[i]
			# w[i] += delta
			for i in range(len(w)):
				delta = ALPHA * (reward + GAMMA * curr_q - prev_q) * prev_state[i]
				w[i] += delta
		print(w)
		for i in range(len(ret)):
			ret[i] += w[i]
	# print(ret)
	for i in range(len(ret)):
		ret[i] /= epoch
	# print(ret)
	ret = [float("{0:.5f}".format(w)) for w in ret]
	return ret


def qFunc(state, weights):
	s = np.asarray([state])
	w = np.asarray([weights]).T
	q = np.matmul(s, w)[0][0]
	return q


if __name__ == '__main__':
	for i in range(30):
		with open("record.txt", "w") as f:
		    pass
		argv = ["-r", "myTeam.py", "-b", "myTeam.py", "-l", "RANDOM", "-n", "2", "-q"]
		options = readCommand(argv)
		runGames(**options)
		argv = ["-r", "myTeam.py", "-b", "shibaReflex.py", "-l", "RANDOM", "-n", "2", "-q"]
		options = readCommand(argv)
		runGames(**options)
		argv = ["-r", "myTeam.py", "-b", "yellowdogReflex.py", "-l", "RANDOM", "-n", "2", "-q"]
		options = readCommand(argv)
		runGames(**options)
		records = readRecords("record.txt")
		with open("weights.txt", "r") as f:
			weights = ast.literal_eval(f.read())
		print(weights)
		new_weights = train(10, records, weights)
		print(new_weights)
		with open("weights.txt", "w") as f:
			f.write(json.dumps(new_weights))
		with open("weights_history.txt", "a") as f:
			f.write(json.dumps(new_weights) + "\n")















