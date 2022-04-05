import numpy as np
import torch
from collections import Counter, defaultdict
from geometry import Point
from world import World
from typing import Union, Optional
from intersection_world import IntersectionScenario
from tqdm import tqdm


def run_intersection_world(data_collection=False):
    states = []
    actions = []
    # instantiate world
    visualize = False if data_collection else True
    wenv = IntersectionScenario(visualize=visualize)
    if not data_collection:
        wenv.render()
    o, d = wenv.reset(), False
    while not d:
        a = None
        o, r, d, info = wenv.step(a)
        if data_collection:
            state, action = info["previous_state_action"]
            states.append(state)
            actions.append(action)
        else:
            wenv.render()

    return np.array(states), np.array(actions)

def collect_data_for_imititation_learning(experiment_method, num_runs=50, outfilepath=None):
    state_data = []
    action_data = []
    print("beginning data collection...")
    for _ in tqdm(range(num_runs)):
        states, actions = experiment_method(data_collection=True)
        state_data.extend(states)
        action_data.extend(actions)
    statefilepath = "{}_states".format(outfilepath)
    actionfilepath = "{}_actions".format(outfilepath)
    with open(statefilepath, "wb") as state_outfle:
        np.save(state_outfle, state_data)
        print("saved data to {}".format(statefilepath))
    with open(actionfilepath, "wb") as action_outfle:
        np.save(action_outfle, action_data)
        print("saved data to {}".format(statefilepath))

collect_data_for_imititation_learning(experiment_method=run_intersection_world, num_runs=50, outfilepath="intersectionsimple_train")