import json, ast
import numpy as np
import random
import util

GAMMA = 0.99
ALPHA = 0.00001


def readRecords(filename):
    records = []
    with open(filename, "r") as f:
        for line in f.readlines():
            record = ast.literal_eval(line)
            records.append(record)
    return records


def train(epoch, records, weights):
    for e in range(epoch):
        # shuffle records
        random.shuffle(records)
        max_update = np.NINF
        temp_weights = weights.copy()
        for r in records:
            p = ast.literal_eval(r["pre"])
            c = ast.literal_eval(r["cur"])
            prev_state = util.Counter(p)
            curr_state = util.Counter(c)
            reward = float(r["r"])
            prev_q = qFunc(prev_state, weights)
            curr_q = qFunc(curr_state, weights)
            for key in weights.keys():
                delta = ALPHA * (reward + GAMMA * curr_q - prev_q) * prev_state[key]
                temp_weights[key] += delta
                # weights[i] = min(10.0, weights[i])
                # weights[i] = max(-10.0, weights[i])
                max_update = max(max_update, abs(delta))
        weights = temp_weights
        # if max_update < 1:
        # 	print("converge", e)
        weights.normalize()
    weights_str = json.dumps(weights)
    return weights_str


def qFunc(state, weights):
    q = state * weights
    return q

def offensiveWeight():
    with open("/home/sakuya/ramdisk/weights1.txt", "r") as f:
        w = ast.literal_eval(f.read())
    return util.Counter(w)

def defensiveWeight():
    with open("/home/sakuya/ramdisk/weights3.txt", "r") as f:
        w = ast.literal_eval(f.read())
    return util.Counter(w)

if __name__ == '__main__':
    records1 = readRecords("/home/sakuya/ramdisk/record1.txt")
    records3 = readRecords("/home/sakuya/ramdisk/record3.txt")
    weight1 = offensiveWeight()
    weight3 = defensiveWeight()
    w1_str = train(20, records1, weight1)
    w3_str = train(20, records3, weight3)
    with open("/home/sakuya/ramdisk/weights1.txt", "w+") as f:
        f.write(w1_str)
    with open("/home/sakuya/ramdisk/weights3.txt", "w+") as f:
        f.write(w3_str)

'''
notes
use better algorithms to estimate enemy's location
add capsule mode?
'''
