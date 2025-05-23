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

# so plan is:
#1. develop nodes that either explore randomly or explore properly. Given that the depth of a node determines
# the type of exploration that you get, it makes sense that we assign that on creation and it is permanent.

# Lots to do! All easy!


