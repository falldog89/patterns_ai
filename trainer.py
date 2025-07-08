""" trainer which takes the replay buffer, samples states, batch trains, and
returns the improved network, along with checkpointing and saving states etc.


First generations:
whenever you meet a state that is terminal next move (either mate - in - one or all moves lose)
store it and save it to disk. Save the tensor state, as we can now easily recover the game and the
augmentations from this.

so need a function that saves this from agent, which will encounter the moves

Then we train on MSE of predicted result (so -1 to 1 value prediction, where that is win or loss for active player)
Just make sure consistent in terms of player or result for active!

Added to this is Cross Entropy loss for the policy. So the visit counts and the policy give the prior probability
of visiting child nodes. I don't think we start training this for a while, until we have learned something about
the terminal states, at least. Otherwise, exploration and depth is random...


Then later generations, full exploration, where we are training on games that are further and further from the terminal
situation.

First generation is just about getting enough different 1 move away terminals to get good at predicting value
of a game 1 move from termination. This can therefore be quite long

WHENEVER a game is one more from termination, forced, we save it, so we can always retrain on those games.

"""

# so the initial training stage is just playing games until they reach "all children terminal"...
# ideally we can also back this up, to get that if there is only one parent move etc.

# we run this until we collect 10k terminal games, where next move ends in death.

# ideally these all come from different starting positions... we would still like to run this from a
# search tree to take advantage of the look ahead for just that 1 move.

# okay so I think the plan is to start designing something that splits the random and the non-random more firmly.

# Random requires NO neural network. We can have an actual random node class, which simply does the look ahead bit
# but requires NO NN evals...

# then the hard bit... we want to run many trees in series, with parallel evals of the NN. I think this is actually okay
# as we just need to iterate over the trees, storing the tensor states until it is full, then provision back. It really
# shouldn't slow things down too much, and just allows the random ones to do there own thing.

# so each exploration step:
# go into each tree

# get the leaf node

# expand
#   - if random, either create a random policy vector for breadth restriction, or create and choose child
#   - if result, return self
#   - if not random, either return own tensor state for eval or create children and return one of them for tensor eval.
#   - if not random, and 2nd visit, create game, return random breadth restricted child.

# stack the tensor state in a list for eval, stack the leaf node that has it
# OR stack the back propagation if no need for eval (random doesn't store results)

# IF the tensor stack is full, eval and provision the value and policy to the nodes that require it (don't waste
# the time otherwise on 1 or 2)

# then back propagate everything that needs it!

# so the agent should just be saying to everyone:
# EVERYONE GO EXPLORE FOR A STEP:
# which means, get a leaf node, expand it, let me know if you require inference
# then evaluate all the inference required, give it back to everyone,
# then backpropagate the results
# then step everything that is ready to be stepped.

#okay so the root node should have an exploration steps and terminal flag...


# ALSO we can start off just training the base model on value.

"""
plan

1. Build robust training class which will deal with 
    a. the formatting and stacking of training data
    b. loading and sorting data from disk including distance from end.
        can weight differently for further out training
    c. augmenting data
    d. plotting and showing loss and validation
    e. ONLY train on the value to start with, manage this too...
    
2. start training for the random games until looks good. Needs to get good at seeing random games 
    and 1 move from the end stuff.
    
3. Start playing games from further back
"""
from mcts.networks import PatternsNet
import matplotlib.pyplot as plt
from typing import Optional

from game import Patterns
from augmentor import StateAugmentor

import torch
import numpy as np
import pickle
import random


class PatternTrainer:
    def __init__(self,
                 network: PatternsNet,
                 device: torch.device,
                 ):
        """ set up the trainer with a network"""
        self.network = network.float()
        self.network.train()
        self.device = device
        self.network.to(self.device)
        self.losses = []

    def collect_sample(self,
                       saved_games: dict,
                       batch_size: int) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """ return tensors of states, visit counts and results of length batch_size from the location.

        Games are saved according to the final result to try and achieve a balance. This might not be important later on
        as the network gets better, but we expect it to be important early on.
        """
        # Attempt a balanced data set with respect to the terminal state:
        expected_size = batch_size // 3

        # balance the data set:
        win_games = saved_games[1]
        loss_games = saved_games[-1]
        draw_games = saved_games[0]

        # determine the number to sample from each chunk:
        num_draw_samples = min(expected_size, len(draw_games))
        num_loss_samples = min(expected_size, len(loss_games))
        num_win_samples = batch_size - num_draw_samples - num_loss_samples

        # choose random games from each terminal state:
        sample_wins = random.sample(win_games, num_win_samples)
        sample_losses = random.sample(loss_games, num_loss_samples)
        sample_draws = random.sample(draw_games, num_draw_samples)

        # state, visit counts, additional information:
        # save_list = [nod.tensor_state, full_visit_counts, _, flipped_num, nod_result]
        win_states, win_vcs, _, _, _ = zip(*sample_wins)
        loss_states, loss_vcs, _, _, _ = zip(*sample_losses)
        draw_states, draw_vcs, _, _, _ = zip(*sample_draws)

        state_tensor = torch.stack(win_states + loss_states + draw_states)
        vc_tensor = torch.tensor(np.stack(win_vcs + loss_vcs + draw_vcs))
        results_tensor = torch.tensor([1] * num_win_samples
                                      + [-1] * num_loss_samples
                                      + [0] * num_draw_samples)

        return state_tensor, vc_tensor, results_tensor.view(-1, 1)

    def augment_sample(self,
                       states: torch.tensor,
                       visit_counts: torch.tensor,
                       is_augment: bool) -> tuple[torch.tensor, torch.tensor]:
        """ if is_augment, exploit the symmetries by permuting colors, reflecting and rotating
        the board at random.
        """

        if not is_augment:
            return states, visit_counts

        augmented_states = []
        augmented_vcs = []

        for _state, _vc in zip(states, visit_counts):
            _sa = StateAugmentor(state=_state,
                                 visit_counts=_vc,
                                 )

            _sa.full_augment()

            # store the augmented states and the augmented visit counts:
            augmented_states.append(_sa.state)
            augmented_vcs.append(_sa.visit_counts)

        stacked_states = torch.stack(augmented_states)
        stacked_vcs = torch.stack(augmented_vcs)

        return stacked_states, stacked_vcs

    @staticmethod
    def mask_policies(prior_policies: torch.tensor, visit_counts: torch.tensor) -> None:
             # torch.tensor):
        """ mask out all non_zero prior policies: note that we have 1 for all legal actions ,
        in the view of the search tree (so also accounting for breadth restrictions)
        """
        prior_policies[visit_counts == 0] = 0

    def get_targets_predictions(self,
                                states: torch.tensor,
                                visit_counts: torch.tensor,
                                results: torch.tensor,
                                is_augment: bool = True,
                                ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """ Take in the data in the form of states, visit counts and results and
        return the predictions and targets together.

        Exploit symmetries and exploit if requested.
        """
        # no op if is_augment is False:
        states, visit_counts = self.augment_sample(states, visit_counts, is_augment)

        # get the predicted results and the policy vectors for the states considered:
        predicted_results, prior_policies = self.network(states.to(self.device).float())

        # in place masking of the prior policies:
        self.mask_policies(prior_policies, visit_counts)

        # normalize the visit counts to form a prior policy:
        visit_counts = visit_counts / visit_counts.sum(1).view(-1, 1)

        # ensure that the dtype of each tensor is the same:
        return results.float(), predicted_results, visit_counts.float(), prior_policies

    def train(self,
              data_location: str,
              is_augment: bool = False,
              epochs: int = 100,
              batch_size: int = 1024,
              is_include_policy: bool=True,
              is_plot_losses: bool=True,
              learning_rate: float = 3e-4,
              ) -> None:
        """
        Load data from location on disk.

        If is_augment, exploit symmetries to create new states with equivalent win loss but different
        representation

        if is_balance_wins, sample equally from winning and losing positions

        if is_balance_moves, sample equally (roughly?) from the move depth.

        """
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate)

        ax = None

        if is_plot_losses:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim([0., float(epochs)])
            ax.set_ylim([0., 1.])

        saved_games = self.load_data(data_location)

        # set network to training mode:
        self.network.train()

        for _epoch in range(epochs):
            if ((_epoch + 1) % 100) == 0:
                print(f"Current epoch: {_epoch}")
                print()

            # sample state, visit counts and results from the saved games:
            _states, _visit_counts, _results = self.collect_sample(saved_games, batch_size)

            # Augment and set up targets and predictions:
            vtarget, vpredictions, ptarget, ppredictions = self.get_targets_predictions(_states,
                                                                                        _visit_counts,
                                                                                        _results,
                                                                                        is_augment)

            # loss function is MSE value and CE policy, unless policy is explicitly neglected:
            loss = self.loss_function(vtarget, vpredictions, ptarget, ppredictions, is_include_policy)

            # backprop steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            self.plot_losses(ax)

    @staticmethod
    def load_data(data_location: str) -> dict:
        """ load the data from disk and return the states, visit counts and final results according to the batch size

        If is_augment, also pass each tensor state through a random augmentation which will rotate, flip and
        permute colors. The action space for the visit counts will also require the same augmentation
        """
        # open data in one go, else too costly to load from disk:
        with open(data_location, mode='rb') as f:
            saved_games = pickle.load(f)

        return saved_games

    def loss_function(self,
                      vtarget: torch.tensor,
                      vpredictions: torch.tensor,
                      ptarget: torch.tensor,
                      ppredictions: torch.tensor,
                      is_include_policy=True) -> torch.nn.Module:
        """ the loss function for the alpha zero style training. Cross entropy for policy,
        added to MSE for target

        vpredictions and ppredictions have gradients from the generation of these from the network,
        vtarget and ptarget are visit counts and results from the MCTS, and do not have gradients.

        """
        loss = torch.nn.MSELoss()(vpredictions, vtarget.to(self.device))

        if is_include_policy:
            policy_loss = torch.nn.CrossEntropyLoss()(ppredictions, ptarget.to(self.device))
            loss += policy_loss

        return loss

    def plot_losses(self, ax: Optional[plt.axes]) -> None:
        """ update the axes to show the latest loss

        Too simple to require own function atm but could be handy when extending:
        """
        if ax is None:
            return

        ax.plot(self.losses)

# todo next step:
    # investigate WHICH ones it does badly on, and why.
    # WILL IT HELP to introduce visit counts? Lets look a little bit, and think about what is going on
    # try and understand why it cannot get better fitting

    def get_MSE_accuracy(self,
                         game_list: list,
                         num_required: int,
                         batch_size: int=2048,
                         is_augment: bool=True) -> torch.tensor:
        """ determine how accurately the network is predicting the output of the given state
        by comparing the actual and predicted states in hold out data:
        """

        predictions = []
        flipped_nums = []

        while len(predictions) < num_required:
            # collect sample of games:
            sample_games = game_list[:batch_size]

            # get the states from the games:
            _states, _vcs, _, _flipped_num, _ = zip(*sample_games)

            # stack the tensors:
            state_tensor = torch.stack(_states)
            vc_tensor = torch.tensor(np.stack(_vcs))

            # augment the states:
            states, visit_counts = self.augment_sample(state_tensor, vc_tensor, is_augment=is_augment)

            # get the predicted results and the policy vectors for the states considered:
            with torch.no_grad():
                # do not collect the prior policy here:
                value_predictions, _ = self.network(states.to(self.device).float())

            # return the value predictions locally:
            predictions.append(value_predictions.cpu())
            flipped_nums.append(_flipped_num)

            # trim the wins list:
            game_list = game_list[batch_size:]

            if batch_size * num_required >= num_required:
                break

        # turn the list of tensors into a single tensor:
        predictions = torch.stack(predictions)
        number_flipped_tokens = np.stack(flipped_nums)

        return predictions, number_flipped_tokens

    def validate_value(self,
                       data_location: str,
                       num_check: int = 1000,
                       is_augment: bool = True,
                       is_plot: bool = True) -> tuple[float, float, float]:
        """ validate the data on positions to which an answer is known. For example one move win loss draw.

        Validations occur in two steps:

        1. We investigate how accurately the network predicts the results. In particular,
        what is the difference between the predicted results and the actual, in MSE terms

        2. We determine whether the action advocated for by the network is
        """
        # the validation data set only includes games that are a single move away from a loss or a win.
        # If the game was won, there might be multiple acceptable actions, whereas in a 1 move loss, every move
        # will result in a loss so there is no point in value-ing the moves.
        validation_games = self.load_data(data_location)

        # set network to eval mode:
        self.network.eval()

        # split the data set:
        wins = validation_games[1]
        losses = validation_games[-1]
        draws = validation_games[0]

        win_predictions, win_length = self.get_MSE_accuracy(wins, num_check, is_augment=is_augment)
        loss_predictions, loss_length = self.get_MSE_accuracy(losses, num_check, is_augment=is_augment)
        draw_predictions, draw_length = self.get_MSE_accuracy(draws, num_check, is_augment=is_augment)

        # win predictions should be
        win_accuracy = ((( win_predictions.numpy() - 1.0) ** 2.0).sum() / len(win_predictions)) ** 0.5
        loss_accuracy = ((( loss_predictions.numpy() + 1.0) ** 2.0).sum() / len(loss_predictions)) ** 0.5
        draw_accuracy = ((( draw_predictions.numpy() - 0.0) ** 2.0).sum() / len(draw_predictions)) ** 0.5

        if is_plot:
            fig, ax = plt.subplots()
            ax.scatter(win_length, win_predictions)
            ax.scatter(loss_length, loss_predictions)
            ax.scatter(draw_length, draw_predictions)

        return win_accuracy, loss_accuracy, draw_accuracy



        #
        # # todo
        # # don't forget to set the network to eval mode in the getting targets predictions bit.:
        # self.network.eval()
        #
        # # determine the predicted and actual results of the validation data:
        # actual_results, predicted_results, _, predicted_actions  = self.get_targets_predictions(validation_data,
        #                                                                                         batch_size,
        #                                                                                         is_augment)
        #
        # # determine whether the predicted actions weight towards the correct result:
        # policy_accuracy = self.check_predicted_actions(validation_data, predicted_actions, predicted_results)
        #
        # # extra wrangling to make sure that equivalently good moves are given equal priority: ie if there are multiple ways
        # # of winning, you don't penalise not taking the "right" one:
        #
        # # MSE for correct value, between the actual results and the predicted value of the 1-move terminal states:
        # correct_value = np.sum((actual_results - predicted_results) ** 2.) ** 0.5
        #
        # # How often was the 1 move win predicted by the policy. Do not look for 1 move losses:
        #
        # # hmmm. So the target data is the child visit counts from the state given in the data. But, when a winning move
        # # is predicted, the step is cancelled after one additional explore step.
        # #
        # # We don't really care about the target move. We care about the predicted argmax producing a winning state.
        #
        #
        # # mask out illegal or unconsidered moves: actual moves here will be the visit counts.
        # max_args = predicted_moves[actual_moves > 0]
        #
        #
        # predictions

    def check_predicted_actions(self,
                                states: torch.tensor,
                                predicted_actions: torch.tensor,
                                results: torch.tensor) -> list[bool]:
        """ take the maximal argument from each prior policy prediction to determine whether
        the highest weight prediction will acheive the correct result
        """
        # NOTE that if we are checking these, we are only checking one move wins and draws.
        # That assumption is baked in:
        accuracy = []

        for _state, _prior_policy, _result in zip(states, predicted_actions, results):
            # ignore non-winning results:
            if _result == -1:
                continue

            # create a new game to test the prior policy:
            _augmentor = StateAugmentor(_state)
            _temp_game = _augmentor.create_game_from_state()

            _predicted_best_action = np.argmax(_prior_policy)
            _, _predicted_result = _temp_game.step(_predicted_best_action)

            # if the predicted result matches the actual result:
            accuracy.append(_predicted_result == _result)

        return accuracy

