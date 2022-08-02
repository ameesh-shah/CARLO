from os import sep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from geometry import Point
from world import World
from typing import Union, Optional
from intersection_world import IntersectionScenario, get_cautious_ego_control
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pickle as pkl
import random

# TODO allow user to choose device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class ImitationLearningNetwork(nn.Module):

    def __init__(self, input_size, output_size, num_units):
        super(ImitationLearningNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_units
        self.first_layer = nn.Linear(self.input_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, current_input):
        assert isinstance(current_input, torch.Tensor)
        current_input = current_input.to(device)
        current = F.relu(self.first_layer(current_input))
        current = self.out_layer(current)
        return current

def create_minibatches(observations, actions, batch_size):
    assert len(observations) == len(actions)
    num_items = len(observations)
    obs_batches = []
    act_batches = []
    def create_single_minibatch(idxseq):
        curr_batch_obs = []
        curr_batch_act = []
        for idx in idxseq:
            curr_batch_obs.append((observations[idx]))
            curr_batch_act.append((actions[idx]))
        return curr_batch_obs, curr_batch_act
    item_idxs = list(range(num_items))
    while len(item_idxs) > 0:
        if len(item_idxs) <= batch_size:
            batch_obs, batch_act = create_single_minibatch(item_idxs)
            obs_batches.append(batch_obs)
            act_batches.append(batch_act)
            item_idxs = []
        else:
            # get batch indices
            batchidxs = []
            while len(batchidxs) < batch_size:
                rando = random.randrange(len(item_idxs))
                index = item_idxs.pop(rando)
                batchidxs.append(index)
            batch_obs, batch_act = create_single_minibatch(batchidxs)
            obs_batches.append(batch_obs)
            act_batches.append(batch_act)
    return obs_batches, act_batches

def train_imitation_nn(observation_filepath: str, action_filepath: str, model: nn.Module, num_epochs,
                       valid_observation_filepath: str, valid_action_filepath: str,
                       lr=0.01, print_every=200, batch_size=1000, model_filepath="saved_nn_model"):
    # set up training configuration
    curr_optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossfxn = nn.MSELoss()
    observation_array = np.load(observation_filepath)
    action_array = np.load(action_filepath)
    print("Creating batches...")
    obs_batches, act_batches = create_minibatches(observation_array, action_array, batch_size=batch_size)
    for epoch in range(1, num_epochs + 1):
        for batchidx in range(len(obs_batches)):
            obs_batch, act_batch = obs_batches[batchidx], act_batches[batchidx]
            true_actions = torch.tensor(act_batch).float().to(device)
            predicted_actions = model.forward(torch.tensor(obs_batch).float())
            #breakpoint()
            loss = lossfxn(predicted_actions, true_actions)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()
            if batchidx % print_every == 0 or batchidx == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))
    # save the model
    print("Saving trained NN model.")
    torch.save(model.state_dict(), model_filepath)
    # evaluate on validation set
    valid_observation_array = np.load(valid_observation_filepath)
    valid_action_array = np.load(valid_action_filepath)
    with torch.no_grad():
        true_actions = torch.tensor(valid_action_array).float().to(device)
        predicted_actions = model.forward(torch.tensor(valid_observation_array).float())
        loss = lossfxn(predicted_actions, true_actions)
        print("validation loss is: {}".format(loss))
    return model

def default_policy_behavior(observation):
    #default in CARLO environment
    return 1


def run_intersection_world(policy_behavior=default_policy_behavior, data_collection=False, separate_controls=False):
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
        a = policy_behavior(o)
        o, r, d, info = wenv.step(a)
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


def collect_data_for_imitation_learning(experiment_method, policy_behavior=default_policy_behavior, num_runs=50, outfilepath=None, separate_controls=False):
    state_data = []
    if separate_controls:
        steer_action_data = []
        accel_action_data = []
    else:
        action_data = []
    if outfilepath is None:
        outfilepath = "trialrun"
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

def learn_regression_tree(observation_filepath, action_filepath, val_observation_filepath, val_action_filepath,
                          rtree_filepath=None):
    if rtree_filepath is None:
        rtree_filepath = "learned_regression_tree"
    observation_array = np.load(observation_filepath)
    action_array = np.load(action_filepath)
    breakpoint()
    reg_tree = DecisionTreeRegressor()
    print("Learning Decision Tree...")
    reg_tree.fit(observation_array, action_array)
    print("Completed Training. Saving regression tree:")
    pkl.dump(reg_tree, open(rtree_filepath, "wb"))
    rmse = deploy_regression_tree(val_observation_filepath, val_action_filepath, reg_tree)
    return reg_tree


def deploy_regression_tree(val_observation_filepath, val_action_filepath, reg_tree):
    print("Analyzing learned model on validation set:")
    validation_observations = np.load(val_observation_filepath)
    predicted_val_actions = reg_tree.predict(validation_observations)
    validation_actions = np.load(val_action_filepath)
    rmse = np.sqrt(((predicted_val_actions - validation_actions) ** 2).mean())
    print("average RMSE is: {}".format(rmse / len(validation_actions)))
    return rmse

class RegressionTreeIntersectionWrapper:

    def __init__(self, rtree_filepath):
        self.rtree_model = pkl.load(open(rtree_filepath, "rb"))

    def make_prediction(self, observation):
        action_prediction = self.rtree_model.predict([observation])[0]
        return np.array([0, action_prediction])

class ImitationNNIntersectionWrapper:

    def __init__(self, imitation_statedict_filepath):
        self.nn_model = ImitationLearningNetwork(17, 1, 8)
        self.nn_model.load_state_dict(torch.load(imitation_statedict_filepath))

    def make_prediction(self, observation):
        with torch.no_grad():
            action_prediction = self.nn_model.forward(torch.tensor(observation).float()).item()
            return np.array([0, action_prediction])

if __name__ == '__main__':
    run_intersection_world(policy_behavior=get_cautious_ego_control)
    # collect_data_for_imitation_learning(run_intersection_world, policy_behavior=get_cautious_ego_control,
    #                                      num_runs=5000, outfilepath="intersection_vals", separate_controls=True)
    # learn_regression_tree("intersection_vals_states", "intersection_vals_accel_actions",
    #                       "intersection_test_states", "intersection_test_steer_actions")
    # reg_tree_wrapper = RegressionTreeIntersectionWrapper("learned_regression_tree")
    # run_intersection_world(policy_behavior=reg_tree_wrapper.make_prediction)
    # nnmodel = ImitationLearningNetwork(17, 1, 8)
    # nnmodel = train_imitation_nn("intersection_vals_states", "intersection_vals_accel_actions",
    #                              nnmodel, 20, "intersection_test_states", "intersection_test_steer_actions",
    #                              )
    nn_wrapper = ImitationNNIntersectionWrapper("saved_nn_model")
    run_intersection_world(policy_behavior=nn_wrapper.make_prediction)
