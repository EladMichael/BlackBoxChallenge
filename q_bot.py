import interface as bbox
import theano
import numpy as np
import random as rand
import theano.tensor as T
import lasagne
import time

epochs = 1
memtime = 10

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(memtime,n_f+2),input_var=in_state)
    net = lasagne.layers.dropout(net,p=.1)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=n_a,nonlinearity=lasagne.nonlinearities.softmax)
    return net

n_f = n_a = max_time = -1
 
def prepare_bbox():
    global n_f, n_a, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_f = bbox.get_num_of_features()
        n_a = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
def forget(states):
    for row in range(len(states)-1,0,-1):
        states[row]=states[row-1]
    states[0] = np.zeros(shape=(1,n_f+2))
    return states

def run_bbox(verbose=False):
    prepare_bbox()

    # vector of the current state features
    input_var= T.matrix('memory')
    input_var= T.reshape(input_var,(memtime,n_f+2))

    #Score after the agent makes it's choice
    reality = T.scalar('consequence')

    #Load net into the agent object
    agent=prepare_agent(input_var)

    #What the agent thinks the best choice will be
    attempt = T.max(lasagne.layers.get_output(agent))

    #how much the agent should be rewarded/punished
    reward = lasagne.objectives.squared_error(attempt,reality)

    #get the parameters for updating
    params = lasagne.layers.get_all_params(agent,trainable=True)

    #update the net with the error
    teach = lasagne.updates.nesterov_momentum(reward,params,learning_rate=0.1,momentum=0.9)

    #function to do all of the stuff above I DON'T HAVE A TARGET??
    train_fn = theano.function([input_var,reality], reward, updates=teach,on_unused_input='ignore')

    # time to check how long it takes to run
    memory = np.zeros(shape=(memtime,n_f+2))
    start = time.time()
    scores_per_epoch = np.zeros(epochs)
    for epoch in range(epochs):
        e_time = time.time() #time for this epoch
        has_next = 1 #looping variable, state of bbox

        #initialize tracking variables
        consequence=0
        self_assessment=0
        steps=0
        trust=0.00
        while has_next:

            #Updating memory matrix, forgetting a state, making room
            memory = forget(memory) 
            state = bbox.get_state()
            #upload new state, with no score or action chosen
            memory[0][:-2] = state
            if rand.random>trust:
                action = rand.randint(0,n_a-1) #if trust is too low still, random action
            else:
                choices = lasagne.get_output(agent,memory) #Otherwise, let the agent decide. 
                action = np.argmax(choices) #pick action agent thinks is best
            
            #do it, and find out the consequences (if the score improved or went down)
            has_next = bbox.do_action(action)
            consequence = bbox.get_score()-consequence 
            
            #train on choices just made and memory
            memory[0][-2:]=[action,consequence]
            train_fn(memory,consequence) #train based on the score change
            
            #updating for next loop
            self_assessment += consequence
            steps += 1

            #occasionally check in on progress
            if steps%10000==0:
                trust = trust+.01
                score = bbox.get_score()
                print ("Epoch: {}".format(epoch))
                print ("Steps: {}".format(steps))
                print ("   self assessment: {}".format(self_assessment))
                print ("   trust: {}".format(trust))
                print ("   current score: {}".format(score))
        #report on model quality on previous epoch
        score = bbox.get_score()
        print ("Epoch: {}".format(epoch))
        print ("Final Score: {}".format(score))
        print ("Time to Run: {} minutes".format((time.time()-e_time)/60))
        scores_per_epoch[epoch] = score

        #reset box for next epoch
        bbox.reset_level()

    print ("All scores per epoch: ")
    print (scores_per_epoch)
    print ("Time to run: {} hours".format((time.time()-start)/3600))
    np.savez('model_mem.npz', *lasagne.layers.get_all_param_values(agent))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 