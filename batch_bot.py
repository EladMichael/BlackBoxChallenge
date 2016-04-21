import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne
import time

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(16,n_features),input_var=in_state)
    net = lasagne.layers.dropout(net,p=.1)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=n_actions,nonlinearity=lasagne.nonlinearities.softmax)
    return net

def load_dataset(name='Data.txt'):
    k=0
    with open(name) as f:
        for line in f:
            k+=1
    states = np.zeros((k,n_features))
    scores = np.zeros((k,n_actions))
    f = open(name,'r')
    for n in range(k):
        line = f.readline()
        print line
        line = line.split()
        print line
        states[n] = float(line[:n_features])
        scores[n] = float(line[n_features:])
    return states,scores,k


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
    # vector of the current state features
    input_var= T.matrix('in_state')
    input_var= T.reshape(input_var,(1000,n_features))
    #vector of the scores for 100 of the same action
    target_var = T.matrix('scores')
    target_var = T.reshape(target_var,(1000,n_actions))
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
    teach = lasagne.updates.nesterov_momentum(punish,params,learning_rate=0.001,momentum=0.9)
    #function to do all of the stuff above
    train_fn = theano.function([input_var, target_var], punish, updates=teach,on_unused_input='ignore')
    # time to check how long it takes to run
    start = time.time()

    states, scores, loops = load_dataset('Full.txt')
    for n in range(loops):
        error=0
        steps=0
        ins = states[n:n+15]
        out = scores[n:n+15]
        action = np.argmax(out[0])
        error = train_fn(ins,out)
        if n%10000==0:
            score = bbox.get_score()
            print ("Steps: {}".format(steps))
            print ("   training loss: {}".format(error))
            print ("   current score: {}".format(score))
        has_next = bbox.do_action(action)
 
    print ("Time to run: {} seconds".format(time.time()-start))
    np.savez('model.npz', *lasagne.layers.get_all_param_values(agent))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 