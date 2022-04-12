from os import sep
import numpy as np
import torch
from collections import Counter, defaultdict
from geometry import Point
from world import World
from typing import Union, Optional
from intersection_world import IntersectionScenario
from tqdm import tqdm


def run_intersection_world(data_collection=False, policy_behavior=1, separate_controls=False):
    states = []
    if separate_controls:
        accel_actions = []
        steer_actions = []
    else:
        actions = []
    # instantiate world
    visualize = False if data_collection else True
    wenv = IntersectionScenario(visualize=visualize)
    if not data_collection:
        wenv.render()
    o, d = wenv.reset(), False
    while not d:
        a = policy_behavior
        o, r, d, info = wenv.step(policy_behavior)
        if data_collection:
            state, action = info["previous_state_action"]
            states.append(state)
            if separate_controls:
                steering, acceleration = action[0], action[1]
                accel_actions.append(np.array([acceleration]))
                steer_actions.append(np.array([steering]))
            else:
                actions.append(action)
        else:
            wenv.render()
    if separate_controls:
        return np.array(states), np.array(steer_actions), np.array(accel_actions)
    else:
        return np.array(states), np.array(actions)

def collect_data_for_imititation_learning(experiment_method, num_runs=50, outfilepath=None, policy_behavior=1, separate_controls=False):
    state_data = []
    if separate_controls:
        steer_action_data = []
        accel_action_data = []
    else:
        action_data = []
    if outfilepath is None:
        outfilepath + "trialrun"
    print("beginning data collection...")
    for _ in tqdm(range(num_runs)):
        datatuple = experiment_method(data_collection=True, policy_behavior=policy_behavior, separate_controls=separate_controls)
        if separate_controls:
            states, steers, accels = datatuple
            state_data.extend(states)
            steer_action_data.extend(steers)
            accel_action_data.extend(accels)
        else:
            states, actions = datatuple
            state_data.extend(states)
            action_data.extend(actions)
    statefilepath = "{}_states".format(outfilepath)
    with open(statefilepath, "wb") as state_outfle:
        np.save(state_outfle, state_data)
        print("saved data to {}".format(statefilepath))
    if separate_controls:
        steeractionfilepath = "{}_steer_actions".format(outfilepath)
        with open(steeractionfilepath, "wb") as s_action_outfle:
            np.save(s_action_outfle, steer_action_data)
            print("saved data to {}".format(steeractionfilepath)) 
        accelactionfilepath = "{}_accel_actions".format(outfilepath)
        with open(accelactionfilepath, "wb") as a_action_outfle:
            np.save(a_action_outfle, accel_action_data)
            print("saved data to {}".format(accelactionfilepath)) 
    else:
        actionfilepath = "{}_actions".format(outfilepath)
        with open(actionfilepath, "wb") as action_outfle:
            np.save(action_outfle, action_data)
            print("saved data to {}".format(actionfilepath))