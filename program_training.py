import numpy as np
import torch
from collections import Counter, defaultdict
from geometry import Point
from world import World
from typing import Union, Optional
from intersection_world import IntersectionScenario
from tqdm import tqdm


def run_intersection_world(data_collection=False):
    state_action_tuples = []
    # instantiate world
    wenv = IntersectionScenario()
    if not data_collection:
        wenv.render()
    o, d = wenv.reset(), False
    while not d:
        a = None
        o, r, d, info = wenv.step(a)
        if data_collection:
            state_action_tuples.append(info["previous_state_action"])
        else:
            wenv.render()

    return np.array(state_action_tuples)

def collect_data_for_imititation_learning(experiment_method, num_runs=500, outfilepath=None):
    data_set = []
    print("beginning data collection...")
    for _ in tqdm(range(num_runs)):
        state_action_tuples = experiment_method(data_collection=True)
        data_set.extend(state_action_tuples)
    with open(outfilepath, "wb") as outfle:
        np.save(outfle, data_set)
        print("saved data to {}".format(outfilepath))

collect_data_for_imititation_learning(experiment_method=run_intersection_world, num_runs=10, outfilepath="intersectionsimple_test")