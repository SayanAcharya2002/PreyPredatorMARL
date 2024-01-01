import torch
torch.set_default_device("cuda:0")
import random
from collections import deque
import numpy as np
from typing import Any

class SimpleLinearNet(torch.nn.Module):
  def __init__(self,inp_size,out_size):
    super().__init__()
    self.lin=torch.nn.Linear(in_features=inp_size,out_features=out_size,dtype=torch.float64)
  
  def forward(self,x):
    if x is None:
      return 0
    return self.lin(x)

class Entity:
  def __init__(self,params):
    self.params=params
    self.replay_buffer=deque(maxlen=params.get('buf_len',100))

    if 'model' in params and 'optim' in params:
      self.brain=params['model']
      self.optimizer=params['optim']
      print("loaded given model")
    elif params['load_from_model_path']:
      self.brain=torch.load(params['model_path'])
      self.optimizer=torch.optim.Adam(self.brain.parameters(),lr=params['lr'])
      print("loaded model from path")
    else:
      self.brain:SimpleLinearNet=None
      self.optimizer=None
      

  @classmethod
  def factory(cls,PreyOrPredClass,n,params):
    ents=[PreyOrPredClass(params)]
    for i in range(n-1):
      ents.append(PreyOrPredClass({
        **params,
        'model':ents[0].brain,
        'optim':ents[0].optimizer
      }))
    
    return ents

  def train_one_step(self,s,a,r,s_,with_replay=True):
    if with_replay:
      self.replay_buffer.append((s,a,r,s_))

    self.optimizer.zero_grad()
    goal=r+(self.params['gamma']*torch.max(self.brain(s_)))
    target=(goal.detach()-self.brain(s)[a])**2
    target.backward()
    self.optimizer.step()

  def choose_action(self,s):
    if self.params['eps']<random.random():
      return random.randint(0,self.params['out_size']-1)
    else:
      q_values=self.brain(s).detach().cpu().numpy()
      q_max=np.max(q_values)    
      return np.random.choice(np.where(q_values==q_max)[0])

class Prey(Entity):
  def __init__(self,params:dict[str,Any]):
    super().__init__(params)
    if not self.brain:
      self.brain=SimpleLinearNet(params['inp_size'],params['out_size'])
      self.optimizer=torch.optim.Adam(self.brain.parameters(),lr=params['lr'])

class Predator(Entity):
  def __init__(self,params:dict[str,Any]):
    super().__init__(params)
    if not self.brain:
      self.brain=SimpleLinearNet(params['inp_size'],params['out_size'])
      self.optimizer=torch.optim.Adam(self.brain.parameters(),lr=params['lr'])


class Environment():
  dirs=np.array(
    [[0,1],
    [0,-1],
    [1,0],
    [-1,0],
    [0,2],
    [0,-2],
    [2,0],
    [-2,0],
    [0,0]],
  )
  def __init__(self,params):
    self.params=params
    self.params_prey=params['prey']
    self.params_pred=params['pred']
    self.pred_vel=np.array(params['pred_vel'])
    self.prey_vel=np.array(params['prey_vel'])

    self.prey_loc=np.random.randint(low=[0,0],high=[params['MAX_X'],params['MAX_Y']],size=(params['prey_c'],2))
    self.preys=Entity.factory(Prey,params['prey_c'],self.params_prey)
    self.pred_loc=np.random.randint(low=[0,0],high=[params['MAX_X'],params['MAX_Y']],size=(params['pred_c'],2))
    self.preds=Entity.factory(Predator,params['pred_c'],self.params_pred)

  def craft_state(self,pos_en,pos_list_op,vel,r):
    state=[]
    state.extend(pos_en)
    state.extend(vel)
    signs=np.array([
      [1,1],
      [-1,1],
      [-1,-1],
      [1,-1],
    ])
    surr=np.zeros((4,))
    for pos_op in pos_list_op:
      rel_pos=pos_op-pos_en
      for ind,sign in enumerate(signs):
        if np.all(rel_pos*sign>=0):
          if np.all(np.abs(rel_pos)<=r):
            surr[ind]=1
    state.extend(surr)

    return torch.tensor(state)

  def step(self,replay_train:bool,step_no):
    prey_vel=self.prey_vel
    pred_vel=self.pred_vel

    #get cur_states and actions
    prey_states1=[]
    prey_actions=[]
    prey_new_locs=[]

    pred_states1=[]
    pred_actions=[]
    pred_new_locs=[]

    for prey_model,prey in zip(self.preys,self.prey_loc):
      prey_states1.append(self.craft_state(prey,self.pred_loc,prey_vel,self.params_prey['r']))
      prey_actions.append(prey_model.choose_action(prey_states1[-1]))
      prey_new_locs.append(prey+Environment.dirs[prey_actions[-1]])

    for pred_model,pred in zip(self.preds,self.pred_loc):
      pred_states1.append(self.craft_state(pred,self.prey_loc,pred_vel,self.params_pred['r']))
      pred_actions.append(pred_model.choose_action(pred_states1[-1]))
      pred_new_locs.append(pred+Environment.dirs[pred_actions[-1]])

    prey_new_locs=np.clip(np.array(prey_new_locs),[0,0],[self.params['MAX_X'],self.params['MAX_Y']])
    pred_new_locs=np.clip(np.array(pred_new_locs),[0,0],[self.params['MAX_X'],self.params['MAX_Y']])
    
    #check validity and train
    put_later_index=[]
    put_later_value=[]
    for i in range(len(self.prey_loc)):
      new_state=self.craft_state(prey_new_locs[i],pred_new_locs,prey_vel,self.params_prey['r'])
      # reward=-10 if torch.all(new_state[-4::]==1) else 1 
      reward=-10 if np.any(np.all(prey_new_locs[i]==pred_new_locs,axis=1)) else 1 
      if reward==-10:
        print(f"prey {i} got eaten at {step_no}")
      self.preys[i].train_one_step(prey_states1[i],prey_actions[i],reward,new_state)
      #reset if caught
      if reward==-10:
        put_later_index.append(i)
        put_later_value.append(np.random.randint(low=[0,0],high=[self.params['MAX_X'],self.params['MAX_Y']],size=(2,)))
        # prey_new_locs[i]=np.random.randint(low=[0,0],high=[self.params['MAX_X'],self.params['MAX_Y']],size=(1,2))
    
    for i in range(len(self.pred_loc)):
      new_state=self.craft_state(pred_new_locs[i],prey_new_locs,pred_vel,self.params_pred['r'])
      # reward=10 if torch.all(new_state[-4::]==1) else -1
      reward=10 if np.sum(np.all(pred_new_locs[i]==prey_new_locs,axis=1)) else -1
      if reward>0:
        print(f"pred {i} ate {reward//10} at {step_no}")
      self.preds[i].train_one_step(pred_states1[i],pred_actions[i],reward,new_state)

    self.prey_loc=prey_new_locs    
    self.pred_loc=pred_new_locs
    if len(put_later_index)>0:
      self.prey_loc[put_later_index]=np.array(put_later_value)

    #replay buffer stuff
    if replay_train:
      for _ in range(self.params['replay_number']):
        rand_prey=random.choice(self.preys)
        rand_prey.train_one_step(*random.choice(rand_prey.replay_buffer),with_replay=False)
        rand_pred=random.choice(self.preds)
        rand_pred.train_one_step(*random.choice(rand_pred.replay_buffer),with_replay=False)
