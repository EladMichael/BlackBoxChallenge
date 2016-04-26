import interface as bbox
import theano
import numpy as np
import random as rand
import theano.tensor as T
import lasagne
import time


epochs = 45
memtime = 10

def prepare_agent(in_state=None):
    net = lasagne.layers.InputLayer(shape=(memtime,n_f+2),input_var=in_state)
    net = lasagne.layers.dropout(net,p=.1)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=50,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.dropout(net,p=.5)
    net = lasagne.layers.DenseLayer(net,num_units=4,nonlinearity=lasagne.nonlinearities.tanh)
    net = lasagne.layers.DenseLayer(net,num_units=n_a,nonlinearity=lasagne.nonlinearities.linear)
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
 

def get_all_score_diffs(state=None,verbose=0):
    initial = bbox.get_score()
    checkpoint_id = bbox.create_checkpoint()
    all_scores = np.zeros(shape=n_a)
    for a in range(n_a):
        for _ in range(100):
            bbox.do_action(a)
        all_scores[a]=bbox.get_score()-initial
        bbox.load_from_checkpoint(checkpoint_id)
    return all_scores

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
    reality = T.vector('score_diffs')

    #Load net into the agent object
    agent=prepare_agent(input_var)

    #What the agent thinks their best choice is this event
    evaluation = lasagne.layers.get_output(agent)[0]

    #how much the agent should be rewarded/punished
    reward = lasagne.objectives.squared_error(evaluation,reality)
    reward = reward.mean()

    #get the parameters for updating
    params = lasagne.layers.get_all_params(agent,trainable=True)

    #update the net with the error
    teach = lasagne.updates.nesterov_momentum(reward,params,learning_rate=0.01,momentum=0.9)

    #A function to get the agent's choice of what to try this time
    decide_fn = theano.function([input_var],evaluation)

    #function to do all of the stuff above
    train_fn = theano.function([input_var,reality], reward, updates=teach,on_unused_input='ignore')

    # time to check how long it takes to run
    start = time.time()
    for epoch in range(epochs):
        memory = np.zeros(shape=(memtime,n_f+2))
        e_time = time.time() #time for this epoch
        has_next = 1 #looping variable, state of bbox
        #initialize tracking variables
        consequence=error=0
        steps=0
        trust=0.00+.02*epoch
        ch=ra=good=0
        while has_next:
            #Updating memory matrix, forgetting a state, making room
            memory = forget(memory) 
            state = bbox.get_state()
            #get best action based on 100 step checkpoint method
            actuals = get_all_score_diffs(state)
            #upload new state, with no score or action chosen
            memory[0][:-2] = state
            if rand.random()>trust:
                action = rand.randint(0,n_a-1) #if trust is too low still, random action
            else:
                choices = decide_fn(memory) #Otherwise, let the agent decide. 
                action = np.argmax(choices) #pick action agent thinks is best

            if action == np.argmax(actuals):
                good = good+1
            #do it, and find out the consequences (if the score improved or went down)
            has_next = bbox.do_action(action)
            #find consequenquence
            score = bbox.get_score()
            consequence=score-consequence
            #train on choices just made and memory
            memory[0][-2:]=[action,consequence]

            error += train_fn(memory,actuals) #train based on the score change

            #updating for next loop
            steps += 1

            #occasionally check in on progress
            if steps%10000==0:
                score = bbox.get_score()
                print ("Epoch: {}".format(epoch))
                print ("Steps: {}".format(steps))
                print ("   current trust: {}".format(trust))
                print ("   avg error: {}".format(error/steps))
                print ("   bad choices: {}%".format(100-float(good)/100))
                print ("   current score: {}".format(score))
                if trust<.95:
                    trust = trust+.02
                bbox.clear_all_checkpoints()
                ch=ra=good=0

        #report on model quality on previous epoch
        score = bbox.get_score()
        with open("epoch_data.txt","a") as f:
        	f.write("Epoch: {}    Final Score: {}    Average Error: {}    Time to Run: {} min\n".format(epoch,score,error/steps,(time.time()-e_time)/60))
        #reset box for next epoch
        np.savez('model_cost.npz', *lasagne.layers.get_all_param_values(agent))
        if(epoch<epochs-1):
            bbox.reset_level()

    print ("Time to run: {} hours".format((time.time()-start)/3600))
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 