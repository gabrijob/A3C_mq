from multiprocessing import Process
from cpython cimport array
import array

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

##########################################################################################
# Global Variable and Hyperparameters
##########################################################################################

OBS_SIZE = 8
CACHE_ACTION_SIZE = 3
FLOW_ACTION_SIZE = 2

net_scope = 'global'
max_global_episodes = 500#5 #500
delay_rate = 4000 # T steps
max_global_steps = 100

GAMMA = 0.999 #0.99
ENTROPY_BETA = 0.1 #0.01
actor_alpha = 0.01   
critic_alpha = 0.01   
actor_hidden = 64#4 #128 #200
critic_hidden = 64#4 #128 #200
N_step = 15


##########################################################################################
# Tensorflow A3C Graph, ops
##########################################################################################
class ACNet(object):
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        OPT_A = tf.compat.v1.train.AdamOptimizer(actor_alpha, beta1=0.99, beta2=0.999, name='OPT_A')
        OPT_C = tf.compat.v1.train.AdamOptimizer(critic_alpha, beta1=0.99, beta2=0.999, name='OPT_C')          
        
        if scope == net_scope: # global
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, OBS_SIZE), name='S')
                # create global net
                self.actor_params, self.critic_params = self._create_net(scope)[-2:] # only require params
                
        else: # local
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, OBS_SIZE), name='S')
                self.a = tf.compat.v1.placeholder(tf.int32, [None, 2], 'A') # shape (batches, [cache_action, flow_action])
                self.critic_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'critic_target')
                self.baselined_returns = tf.compat.v1.placeholder(tf.float32, [None, 1], 'baselined_returns') # for calculating advantage 
                # create local net
                self.action_prob, self.V, self.actor_params, self.critic_params = self._create_net(scope)
                    
                TD_err = tf.subtract(self.critic_target, self.V, name='TD_err')
                with tf.compat.v1.name_scope('actor_loss'):
                    log_prob = tf.reduce_sum(input_tensor=tf.math.log(self.action_prob + 1e-5) * tf.one_hot(self.a, CACHE_ACTION_SIZE+FLOW_ACTION_SIZE, dtype=tf.float32), axis=1, keepdims=True)
                    actor_component = log_prob * tf.stop_gradient(self.baselined_returns)
                    # entropy for exploration
                    entropy = -tf.reduce_sum(input_tensor=self.action_prob * tf.math.log(self.action_prob + 1e-5), axis=1, keepdims=True)  # encourage exploration
                    self.actor_loss = tf.reduce_mean( input_tensor=-(ENTROPY_BETA * entropy + actor_component) )                                        
                with tf.compat.v1.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(input_tensor=tf.square(TD_err))                      
                # accumulated gradients for local actor    
                with tf.compat.v1.name_scope('local_actor_grad'):                   
                    self.actor_zero_op, self.actor_accumu_op, self.actor_apply_op, actor_accum = self.accumu_grad(OPT_A, self.actor_loss, scope=scope + '/actor')
                # ********** accumulated gradients for local critic **********
                with tf.compat.v1.name_scope('local_critic_grad'):
                    self.critic_zero_op, self.critic_accumu_op, self.critic_apply_op, critic_accum = self.accumu_grad(OPT_C, self.critic_loss, scope=scope + '/critic')
                    
            with tf.compat.v1.name_scope('params'): # push/pull from local/worker perspective
                with tf.compat.v1.name_scope('push_to_global'):
                    self.push_actor_params = OPT_A.apply_gradients(zip(actor_accum, globalAC.actor_params))
                    self.push_critic_params = OPT_C.apply_gradients(zip(critic_accum, globalAC.critic_params))
                with tf.compat.v1.name_scope('pull_fr_global'):
                    self.pull_actor_params = [local_params.assign(global_params) for local_params, global_params in zip(self.actor_params, globalAC.actor_params)]
                    self.pull_critic_params = [local_params.assign(global_params) for local_params, global_params in zip(self.critic_params, globalAC.critic_params)]                    
                    
    def _create_net(self, scope):
        w_init = tf.compat.v1.glorot_uniform_initializer()
        with tf.compat.v1.variable_scope('actor'):
            hidden = tf.compat.v1.layers.dense(self.s, actor_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')
            action_prob = tf.compat.v1.layers.dense(hidden, CACHE_ACTION_SIZE+FLOW_ACTION_SIZE, tf.nn.softmax, kernel_initializer=w_init, name='action_prob') 
            # action_prob output is an array of size 5, so a shape correction is needed to get the composite action tuple       
        with tf.compat.v1.variable_scope('critic'):
            hidden = tf.compat.v1.layers.dense(self.s, critic_hidden, tf.nn.relu6, kernel_initializer=w_init, name='hidden')
            V = tf.compat.v1.layers.dense(hidden, 1, kernel_initializer=w_init, name='V')         
        actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')       
        return action_prob, V, actor_params, critic_params

    def accumu_grad(self, OPT, loss, scope):
        # retrieve trainable variables in scope of graph
        #tvs = tf.trainable_variables(scope=scope + '/actor')
        tvs = tf.compat.v1.trainable_variables(scope=scope)
        # ceate a list of variables with the same shape as the trainable
        accumu = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        zero_op = [tv.assign(tf.zeros_like(tv)) for tv in accumu] # initialized with 0s
        gvs = OPT.compute_gradients(loss, tvs) # obtain list of gradients & variables
        #gvs = [(tf.where( tf.is_nan(grad), tf.zeros_like(grad), grad ), var) for grad, var in gvs]
        # adds to each element from the list you initialized earlier with zeros its gradient 
        # accumu and gvs are in same shape, index 0 is grads, index 1 is vars
        accumu_op = [accumu[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        apply_op = OPT.apply_gradients([(accumu[i], gv[1]) for i, gv in enumerate(gvs)]) # apply grads
        return zero_op, accumu_op, apply_op, accumu      
      
    def push_global_actor(self, feed_dict): 
        SESS = self.sess
        SESS.run([self.push_actor_params], feed_dict)  

    def push_global_critic(self, feed_dict):  
        SESS = self.sess
        SESS.run([self.push_critic_params], feed_dict)         
        
    def pull_global(self):  
        SESS = self.sess
        SESS.run([self.pull_actor_params, self.pull_critic_params])

    def choose_action(self, s):  
        SESS = self.sess
        prob_weights = SESS.run(self.action_prob, feed_dict={self.s: s[None,:]})
        ## divide prob_weights and get action1 and action2 and return tuple
        cache_action = np.random.choice(range(prob_weights.shape[1])[:CACHE_ACTION_SIZE], p=prob_weights.ravel()[:CACHE_ACTION_SIZE])
        flow_action = np.random.choice(range(prob_weights.shape[1])[FLOW_ACTION_SIZE:], p=prob_weights.ravel()[FLOW_ACTION_SIZE:])
        #print("prob_weights ")
        #print(prob_weights)
        return (cache_action, flow_action)            

    def init_grad_storage_actor(self):
        SESS = self.sess
        SESS.run(self.actor_zero_op)
        
    def accumu_grad_actor(self, feed_dict):
        SESS = self.sess
        SESS.run([self.actor_accumu_op], feed_dict)          
    
    def apply_accumu_grad_actor(self, feed_dict):
        SESS = self.sess
        SESS.run([self.actor_apply_op], feed_dict)   
        
    def init_grad_storage_critic(self):
        SESS = self.sess
        SESS.run(self.critic_zero_op)
        
    def accumu_grad_critic(self, feed_dict):
        SESS = self.sess
        SESS.run([self.critic_accumu_op], feed_dict)          
    
    def apply_accumu_grad_critic(self, feed_dict):
        SESS = self.sess
        SESS.run([self.critic_apply_op], feed_dict)

##########################################################################################
# Environment Class
##########################################################################################
class SparkMQEnv(object): # local only
    def __init__(self, upper_limit=50, lower_limit=1):
        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

        self.maxqos = upper_limit
        self.minqos = lower_limit

    def calc_reward(self, state):
        thpt_avg = state[0]
        max_cache = state[7]
        sched_t = state [3]

        reward = - max_cache/thpt_avg - sched_t

        return reward 


##########################################################################################
# Worker Class
##########################################################################################
class Worker(object): # local only
    def __init__(self, name, globalAC, GLOBAL_EP, GLOBAL_RUNNING_R, sess, state):
        self.env = SparkMQEnv()
        self.name = name
        self.AC = ACNet(name, sess, globalAC)
        self.sess = sess
        self.GLOBAL_EP = GLOBAL_EP
        self.GLOBAL_RUNNING_R = GLOBAL_RUNNING_R
        self.T = 0
        self.t = 0
        self.s = state
        self.start_episode()

    def change_state(self, state, done):
        #s_, r, done, info = self.env.step(a)
        self.done = done
        self.r = self.env.calc_reward(state)
        self.ep_r += self.r
        self.buffer_s.append(self.s)
        self.buffer_r.append(self.r)
        self.buffer_done.append(self.done)                
        self.s = np.array(state)
        self.t += 1

        if self.t > max_global_steps:
            self.end_episode()
            self.start_episode()

    def get_action(self):
        composite_action = self.AC.choose_action(self.s)
        self.buffer_a.append(composite_action)
        return composite_action

    def end_episode(self):
        # if statement will always be done in this case... 
        # possible future modification
        if self.done:
            V_s = 0   
        else:
            V_s = self.sess.run(self.AC.V, {self.AC.s: self.s[None, :]})[0, 0] # takes in just one s, not a batch.
            
        # critic related
        critic_target = self.discount_rewards(self.buffer_r, GAMMA, V_s)
            
        buffer_s, buffer_a, critic_target = np.vstack(self.buffer_s), np.array(self.buffer_a), np.vstack(critic_target)
        feed_dict = {self.AC.s: buffer_s, self.AC.critic_target: critic_target}                         
        self.AC.accumu_grad_critic(feed_dict) # accumulating gradients for local critic  
        self.AC.apply_accumu_grad_critic(feed_dict) 
            
        baseline = self.sess.run(self.AC.V, {self.AC.s: buffer_s}) # Value function
        epr = np.vstack(self.buffer_r).astype(np.float32)
        #V_s = SESS.run(self.AC.V, {self.AC.s: s[None, :]})[0, 0] # takes in just one s, not a batch.
        #n_step_targets = self.n_step_targets_missing(epr, baseline, GAMMA, N_step) # Q values
        n_step_targets = self.n_step_targets_max(epr, baseline, V_s, GAMMA, N_step) # Q values
        # Advantage function
        baselined_returns = n_step_targets - baseline

        feed_dict = {self.AC.s: buffer_s, self.AC.a: buffer_a, self.AC.critic_target: critic_target, self.AC.baselined_returns: baselined_returns}            
        self.AC.accumu_grad_actor(feed_dict) # accumulating gradients for local actor  
            
        # update
        self.AC.push_global_actor(feed_dict)                
        self.AC.push_global_critic(feed_dict)
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_done = [], [], [], []
        self.AC.pull_global()
              
        if self.T % delay_rate == 0: # delay clearing of local gradients storage to reduce noise
            # apply to local
            self.AC.init_grad_storage_actor() # initialize storage for accumulated gradients.
            self.AC.init_grad_storage_critic() 
                
        #GLOBAL_EP += 1                   
        self.sess.run(self.GLOBAL_EP.assign_add(1.0))
        #GLOBAL_RUNNING_R.append(ep_r) # for display
        qe = self.GLOBAL_RUNNING_R.enqueue(self.ep_r)
        self.sess.run(qe) 

    def start_episode(self):
        #self.s = self.env.reset()
        self.ep_r = 0 # reward per episode
        self.done = False
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_done = [], [], [], []
        self.AC.pull_global()

            
    def discount_rewards(self, r, gamma, running_add):
      """Take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      #running_add = 0
      for t in reversed(range(len(r))):
          running_add = running_add * gamma + r[t]
          discounted_r[t] = running_add
      return discounted_r 
  
    # As n increase, variance increase.
    # Create a function that returns an array of n-step targets, one for each timestep:
    # target[t] = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^n V(s_{t+n})
    # Where r_t is given by episode reward (epr) and V(s_n) is given by the baselines.
    
    def n_step_targets_missing(self, epr, baselines, gamma, N):
      targets = np.zeros_like(epr)    
      if N > epr.size:
        N = epr.size
      for t in range(epr.size):    
        for n in range(N):
          if t+n == epr.size:            
            break # missing terms treated as 0
          if n == N-1: # last term
            targets[t] += (gamma**n) * baselines[t+n]
          else:
            targets[t] += (gamma**n) * epr[t+n] 
      return targets  
    
    def n_step_targets_max(self, epr, baselines, v_s_, gamma, N):
      targets = np.zeros_like(epr)    
      if N > epr.size:
        N = epr.size
      for t in range(epr.size):  
        #print("t=", t)
        for n in range(N):
          #print("n=", n)
          if t+n == epr.size:            
            targets[t] += (gamma**n) * v_s_ # use max steps available
            break 
          if n == N-1: # last term
            targets[t] += (gamma**n) * baselines[t+n]
          else:
            targets[t] += (gamma**n) * epr[t+n] 
      return targets


##########################################################################################
# Cluster Setup
##########################################################################################
cluster = tf.train.ClusterSpec({
    "worker": ["localhost:2223"],
    "ps": ["localhost:2225"]
})

def parameter_server(max_exec_time):
    #tf.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
 
    server = tf.distribute.Server(cluster,
                             job_name="ps",
                             task_index=0)
    sess = tf.compat.v1.Session(target=server.target)        
    
    with tf.device("/job:ps/task:0"):
        GLOBAL_AC = ACNet(net_scope, sess, globalAC=None) # only need its params
        GLOBAL_EP = tf.Variable(0.0, name='GLOBAL_EP') # num of global episodes   
        # a queue of ep_r
        GLOBAL_RUNNING_R = tf.queue.FIFOQueue(max_global_episodes, tf.float32, shared_name="GLOBAL_RUNNING_R")        
    
    print("Parameter server: waiting for cluster connection...")
    sess.run(tf.compat.v1.report_uninitialized_variables())
    print("Parameter server: cluster ready!")
    
    print("Parameter server: initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("Parameter server: variables initialized")
    
    start_time = time.time()
    while True:
        time.sleep(1.0)
        #print("ps 1 GLOBAL_EP: ", sess.run(GLOBAL_EP))
        #print("ps 1 GLOBAL_RUNNING_R.size(): ", sess.run(GLOBAL_RUNNING_R.size()))  
        #if sess.run(GLOBAL_RUNNING_R.size()) >= max_global_episodes: # GLOBAL_EP starts from 0, hence +1 to max_global_episodes          
        if (time.time() - start_time) >= max_exec_time:
            time.sleep(5.0)
            #print("ps 2 GLOBAL_RUNNING_R.size(): ", sess.run(GLOBAL_RUNNING_R.size()))  
            GLOBAL_RUNNING_R_list = []
            for j in range(sess.run(GLOBAL_RUNNING_R.size())):
                ep_r = sess.run(GLOBAL_RUNNING_R.dequeue())
                GLOBAL_RUNNING_R_list.append(ep_r) # for display
            break
              
    # display
    plt.plot(np.arange(len(GLOBAL_RUNNING_R_list)), GLOBAL_RUNNING_R_list)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()  

    #print("Parameter server: blocking...")
    #server.join() # currently blocks forever    
    print("Parameter server: ended...")

class WorkerAPI(): 
    def __init__(self, worker_n, start_state):
        #tf.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        self.worker_n = worker_n

        self.server = tf.distribute.Server(cluster,
                                job_name="worker",
                                task_index=worker_n)
        self.sess = tf.compat.v1.Session(target=self.server.target)  
    
        with tf.device("/job:ps/task:0"):
            GLOBAL_AC = ACNet(net_scope, self.sess, globalAC=None) # only need its params
            GLOBAL_EP = tf.Variable(0.0, name='GLOBAL_EP') # num of global episodes
            # a queue of ep_r
            GLOBAL_RUNNING_R = tf.queue.FIFOQueue(max_global_episodes, tf.float32, shared_name="GLOBAL_RUNNING_R")   
        """
        with tf.device(tf.train.replica_device_setter(
                            worker_device='/job:worker/task:' + str(worker_n),
                            cluster=cluster)):
        """                        
        print("Worker %d: waiting for cluster connection..." % worker_n)
        self.sess.run(tf.compat.v1.report_uninitialized_variables())
        print("Worker %d: cluster ready!" % worker_n)
        
        #while sess.run(tf.report_uninitialized_variables()):
        while (self.sess.run(tf.compat.v1.report_uninitialized_variables())).any(): # ********** .any() .all() **********
            print("Worker %d: waiting for variable initialization..." % worker_n)
            time.sleep(1.0)
        print("Worker %d: variables initialized" % worker_n)
        
        self.w = Worker(str(worker_n), GLOBAL_AC, GLOBAL_EP, GLOBAL_RUNNING_R, self.sess, start_state) 
        print("Worker %d: created" % worker_n)
        
        self.sess.run(tf.compat.v1.global_variables_initializer()) # got to initialize after Worker creation
    
    def infer(self, state):
        #w.work()
        print("Worker %d: w.work()" % self.worker_n)
        self.w.change_state(state, done=False)

        action = self.w.get_action()

        return action
          
    def finish(self, state):
        self.w.change_state(state, done=True)
        #print("Worker %d: blocking..." % worker_n)
        self.server.join() # currently blocks forever
        print("Worker %d: ended..." % self.worker_n)


##########################################################################################
# Cython API
##########################################################################################

cdef public object create_worker(float* start_state, int worker_n):
    state = []
    for i in range(OBS_SIZE):
        state.append(start_state[i])
    
    return WorkerAPI(worker_n, state)


cdef public int* worker_infer(object agent , float* new_state):
    state = []
    for i in range(OBS_SIZE):
        state.append(new_state[i])

    actions = agent.infer(state)
    cdef int action_arr[2];
    action_arr[0] = actions[0];
    action_arr[1] = actions[1];
    #cdef array.array action_arr = array.array('i', actions)
    #action = action % (ACTION_SPACE_SIZE/2) - 1

    return action_arr

cdef public void worker_finish(object agent, float* last_state):
    state = []
    for i in range(OBS_SIZE):
        state.append(last_state[i])
    
    agent.finish(state)

cdef public object parameter_server_proc(float max_time):
    ps_proc = Process(target=parameter_server, args=(max_time, ), daemon=True)
    ps_proc.start()

    return ps_proc
    # if not join, parent will terminate before children 
    #ps_proc.join() 

    #ps_proc.terminate()

cdef public void parameter_server_kill(object p_server):
    p_server.terminate()
    return

##########################################################################################
# Setup process, rollout and train
##########################################################################################
'''
start_time = time.time()

ps_proc = Process(target=parameter_server, daemon=True)
w1_proc = Process(target=worker, args=(0, ), daemon=True)
w2_proc = Process(target=worker, args=(1, ), daemon=True)

ps_proc.start()
w1_proc.start()
w2_proc.start()

# if not join, parent will terminate before children 
# & children will terminate as well cuz children are daemon
ps_proc.join() 
#w1_proc.join()
#w2_proc.join() 
    
for proc in [w1_proc, w2_proc, ps_proc]:
    proc.terminate() # only way to kill server is to kill it's process
        
print('All done.')     

print("--- %s seconds ---" % (time.time() - start_time))
'''

