import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(1,n_features),input_var=in_state)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=n_actions,nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_all_scores(state=None,verbose=0):
    checkpoint_id = bbox.create_checkpoint()
    all_scores = np.zeros(shape=n_actions)
    for a in range(n_actions):
        bbox.do_action(a)
        all_scores[a]=bbox.get_score()
        bbox.load_from_checkpoint(checkpoint_id)
    return all_scores

def get_action_by_state(state=None):
    scores = get_all_scores(state)
    action_to_do = np.argmax(scores)
    # print (scores,action_to_do)
    # raw_input("Press Enter to continue...")
    return action_to_do


n_features = n_actions = max_time = -1

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    has_next = 1

    prepare_bbox()
    #vector of the current state features
    input_var= T.dvector('in_state')
    input_var= T.reshape(input_var,(1,n_features))
    #vector of the scores for 100 of the same action
    target_var = T.dvector('scores')
    target_var = T.reshape(target_var,(1,n_actions))
    #Load net into the agent object
    agent=prepare_agent(input_var)
    #what the agent thinks will happen if it does each action 100 times
    attempt = lasagne.layers.get_output(agent)
    #how much the agent was wrong, and should be punished
    punish = lasagne.objectives.squared_error(attempt,target_var)
    punish = punish.mean()
    #get the parameters for updating
    params = lasagne.layers.get_all_params(agent,trainable=True)
    #update the net with the error
    teach = lasagne.updates.nesterov_momentum(punish,params,learning_rate=.1,momentum=.9)
    #function to do all of the stuff above
    train_fn = theano.function([input_var, target_var], punish, updates=teach,on_unused_input='ignore')
    #time to check how long it takes to run
    start = time.time()
    while has_next:
        state = bbox.get_state()
        r_state= np.reshape(state,(1,n_features))
        scores = get_all_scores(state)
        r_scores = np.reshape(scores,(1,n_actions))
        action = T.argmax(scores)
        error = train_fn(r_state,r_scores)
        print (error)
        has_next = bbox.do_action(action)
 
    print ("Time to run: {} seconds".format(time.time()-start))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 
