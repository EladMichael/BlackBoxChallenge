import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(1,n_features),input_var=in_state)
    net = lasagne.layers.DenseLayer(net,num_units=300,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=300,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=n_actions,nonlinearity=lasagne.nonlinearities.softmax)
    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)
    return net

# def get_all_score_diffs(state=None,verbose=0):
#     initial = bbox.get_score()
#     checkpoint_id = bbox.create_checkpoint()
#     all_scores = np.zeros(shape=n_actions)
#     for a in range(n_actions):
#         for _ in range(100):
#             bbox.do_action(a)
#         all_scores[a]=bbox.get_score()-initial
#         bbox.load_from_checkpoint(checkpoint_id)
#     return all_scores

# def get_action_by_state(state=None):
#     scores = get_all_score_diffs(state)
#     action_to_do = np.argmax(scores)
#     # print (scores,action_to_do)
#     # raw_input("Press Enter to continue...")
#     return action_to_do


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
    #Load net into the agent object
    agent=prepare_agent(input_var)
    attempt = lasagne.layers.get_output(agent)
    #function to do all of the stuff above
    eval_fn = theano.function([input_var], attempt,on_unused_input='ignore')
    #time to check how long it takes to run
    start = time.time()
    error=0
    steps=0
    while has_next:
        state = bbox.get_state()
        r_state= np.reshape(state,(1,n_features))
        attempt = eval_fn(r_state)
        action = np.argmax(attempt)
        steps+=1
        if steps%10000==0:
            score = bbox.get_score()
            print ("Steps: {}".format(steps))
            print ("   training loss: {}".format(error/steps))
            print ("   current score: {}".format(score))
        has_next = bbox.do_action(action)
 
    print ("Time to run: {} seconds".format(time.time()-start))
    print ("{} steps total".format(steps))
    np.savez('model.npz', *lasagne.layers.get_all_param_values(agent))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 