import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time

epochs = 1
memtime = 15

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(memtime,n_f+2),input_var=in_state)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=4,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=n_a,nonlinearity=lasagne.nonlinearities.linear)
    with np.load('model_mem.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)
    return net

n_f = n_a = max_time = -1

def prepare_bbox():
    global n_f, n_a, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/test_level.data", verbose=1)
        n_f = bbox.get_num_of_features()
        n_a = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()

def forget(states):
    for row in range(len(states)-1,0,-1):
        states[row]=states[row-1]
    states[0] = np.zeros(shape=(1,n_f+2))
    return states

def run_bbox(verbose=False):
    has_next = 1
    prepare_bbox()
    # vector of the current state features

    input_var= T.dvector('in_state')
    input_var= T.reshape(input_var,(memtime,n_f+2))

    #Load net into the agent object
    agent=prepare_agent(input_var)

    #What the agent thinks the best choice will be
    attempt = lasagne.layers.get_output(agent)[0]

    #function to do all of the stuff above
    test_fn = theano.function([input_var], attempt)
    # time to check how long it takes to run
    memory = np.zeros(shape=(memtime,n_f+2))
    start = time.time()
    consequence=0
    steps=0
    while has_next:
        memory = forget(memory)
        state = bbox.get_state()
        memory[0][:-2]=state
        choices = test_fn(memory)
        action = np.argmax(choices)
        has_next = bbox.do_action(action)
        score = bbox.get_score()
        consequence=score-consequence
        memory[0][-2:] = [action,consequence]
        steps+=1
        if steps%10000==0:
            score = bbox.get_score()
            print ("Steps: {}".format(steps))
            print ("   current score: {}".format(score))

    print ("Final Score: {}".format(score))
    print ("Time to run: {} seconds".format(time.time()-start))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 