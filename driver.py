import torch
torch.set_default_device("cuda:0")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
from collections import deque
import numpy as np
import random,time
import utils


MAX_X=100
MAX_Y=100
num_episode=100
max_steps=200
prey_model_path='models/prey.pth'
pred_model_path='models/pred.pth'
load_from_model_path=False
gui_buffer_prey=deque(maxlen=200)
gui_buffer_pred=deque(maxlen=200)



params={
    'MAX_X':MAX_X,
    'MAX_Y':MAX_Y,
    'pred_c':20,
    'prey_c':10,
    'replay_number':40,
    'pred_vel':[5,5],
    'prey_vel':[5,5],
    'prey':{
      'r':20,
      'lr':None,
      'gamma':0.99,
      'inp_size':8,
      'out_size':9,
      'eps':None,
      'model_path':prey_model_path,
      'load_from_model_path':load_from_model_path,
    },
    'pred':{
      'r':10,
      'lr':None,
      'gamma':0.99,
      'inp_size':8,
      'out_size':8,
      'eps':None,
      'model_path':pred_model_path,
      'load_from_model_path':load_from_model_path,
    }
  }

for episode_no in range(num_episode):
  fig=plt.figure()
  axis=plt.axes(xlim=(0,MAX_X),ylim=(0,MAX_Y))

  def anim(i):
    axis.cla()
    axis.set_xlim(0,MAX_X)
    axis.set_ylim(0,MAX_Y)
    # print(gui_buffer_prey[i])
    axis.scatter(gui_buffer_prey[i][:,0],gui_buffer_prey[i][:,1],s=25,c='b',marker='o') 
    axis.scatter(gui_buffer_pred[i][:,0],gui_buffer_pred[i][:,1],s=25,c='r',marker='x')

  print(f"currently at {episode_no}")
  lr=1e-4+(1e-5-1e-4)*episode_no/num_episode
  eps=0.1+(0.1-0.01)*episode_no/num_episode
  for string in ('pred','prey'):
    params[string]['lr']=lr
    params[string]['eps']=eps


  env=utils.Environment(params)
  for step_no in range(max_steps):
    gui_buffer_prey.append(env.prey_loc.copy()) 
    gui_buffer_pred.append(env.pred_loc.copy())
    env.step(replay_train=step_no>10,step_no=step_no)

    if step_no%max_steps==199:
      animation=FuncAnimation(fig,anim,frames=50,interval=100,blit=False,repeat=False)
      plt.show()
      del animation
      pass

  load_from_model_path=True
  params['prey']['model']=env.preys[0].brain
  params['prey']['optim']=env.preys[0].optimizer
  params['pred']['model']=env.preds[0].brain
  params['pred']['optim']=env.preds[0].optimizer
  
  torch.save(env.preys[0].brain,prey_model_path)
  torch.save(env.preds[0].brain,pred_model_path)