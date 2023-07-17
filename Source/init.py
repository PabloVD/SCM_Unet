import random
import numpy as np
import torch
import os

print(torch.__version__)

path=""

simspath = "/export/work/pvillanueva/SCM_Simulations/"
#simspath = "/home/tda/Descargas/SCM_simulations/"

#dataname = "OverfitFlatSettling"
#dataname = "FlatSettling"
#dataname = "DefSims"
dataname = "DefSimsNoDamp"

pathchrono = simspath + dataname + "/Train"
pathchronoadd = simspath + dataname + "/TrainAdd"
pathvalid = simspath + dataname + "/Valid"

# Random seeds
random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

"""
device = torch.device('cpu')
print("\nUSING CPU FOR DEBUGGING!!!\n")
#"""

#--- Global parameters ---#

# Dimension
dim = 3

# Timestep used for training
dt = 0.02   # s

# Arbitrary factor normalization for velocity and acceleration (more stable training)
vel_std = 1.   # m/s
acc_std = 1.   # m/s^2
#vel_std = 0.0008
#acc_std = 2.e-5

# Force factor normalization (more stable training)
#"""
#force_mean = [0., 0., 3400.]
force_mean = [0., 0., 3.5e3]
force_std = [1.e3, 1.e3, 2.e3]
torque_mean = [0., 0., 0.]
torque_std = [350., 350., 180.]
force_min = [-1.e4, -1.8e4, -1.e3]
force_max = [1.e4, 1.8e4, 2.e4]
torque_min = [-4.7e3, -3.5e3, -1.2e3]
torque_max = [4.7e3, 3.5e3, 1.2e3]
print("Using normalized force and torque")
"""
force_mean = [0., 0., 0.]
force_std = [1., 1., 1.]
torque_mean = [0., 0., 0.]
torque_std = [1., 1., 1.]
#"""
force_mean, force_std = torch.tensor(force_mean, device=device, dtype=torch.float32).view(1,3), torch.tensor(force_std, device=device, dtype=torch.float32).view(1,3)
torque_mean, torque_std = torch.tensor(torque_mean, device=device, dtype=torch.float32).view(1,3), torch.tensor(torque_std, device=device, dtype=torch.float32).view(1,3)
force_min, force_max = torch.tensor(force_min, device=device, dtype=torch.float32).view(1,3), torch.tensor(force_max, device=device, dtype=torch.float32).view(1,3)
torque_min, torque_max = torch.tensor(torque_min, device=device, dtype=torch.float32).view(1,3), torch.tensor(torque_max, device=device, dtype=torch.float32).view(1,3)


# Noise standard deviation for regularization
#noise_std = 1.e-2
#noise_std = 5.e-3
noise_std = 1.e-3
#noise_std = 5.e-4
#noise_std = 1.e-4
#noise_std = 0.

# Training sequence: use current timestep plus seq_len-1 previous as input data
#seq_len = 2
seq_len = 1

# Plot loss in tensorboard after log_steps gradient updates
log_steps = 10

# Save model after steps_save_model gradient updates
steps_save_model = 1000

# Rigid body index
indrig = 1

permu = [0,2,1]

# Hyperparameters
num_epochs = 5000
lr_fact = 1.
lr_max = 1.e-4*lr_fact
lr_min = 1.e-7*lr_fact
weight_decay = 1.e-8
#weight_decay = 1.e-5

# Batch size
#batch_size = 20
#batch_size = 40
#batch_size = 60
batch_size = 128
#batch_size = 80
#batch_size = 100


data_aug = True
print("Rotation data augmentation:",data_aug)

# Use torque as rigid output, instead use rotation quaternion
use_torque = True
#use_torque = False

# Use equivariant network
use_equi = True
#use_equi = False

print("Min learning rate: {:.1e}".format(lr_min))
print("Max learning rate: {:.1e}".format(lr_max))
print("Noise std: {:.1e}".format(noise_std))
print("Use torque:", use_torque)
print("Use equivariant net:", use_equi)
print("Batch size:", batch_size)

# Linking radius
linkradius = 0.055
#linkradius = 0.075
# Number of hidden features in each MLP
hidden_channels = 64
# Number of intermediate message passing layers
n_layers = 3

print("Linking radius",linkradius)
print("Hidden channels",hidden_channels)
print("Graph layers",n_layers)

# If true, sample a smaller box of particles (in depth) for training
subsample = True
#subsample = False
depthdist = 0.05

# Polaris wheel geometric parameters
wheel_radius = 0.330229
wheel_semiwidth = 0.2121/2.
# wheel_radius + 2.*r_link
# wheel_width + 2.*r_link

window_x = wheel_radius + 0.15
window_y = wheel_radius + 0.15
window_radius = wheel_radius + 0.1

# If true, use throttle, steering and braking as additional inputs
#use_throttle = True
use_throttle = False
print("Using throttle, steering and braking:",use_throttle)

"""
# Gravity stuff
grav = 9.80665  # m/s²
gravity = torch.tensor([0.,0.,-grav*dt**2.],dtype=torch.float32,device=device).view(1,3)
#"""


use_wheeltype = False
use_wheeltype = True
print("Using wheel type embedding:",use_wheeltype)

#use_boundary = True
use_boundary = False
print("Using bottom boundary distance:",use_boundary)

# 1k nodes fps
#wheelnodesfile = "wheelnodes_scm.npy"
# full nodes
#wheelnodesfile = "wheelnodes_scm_full.npy"
# 2k nodes
#wheelnodesfile = "wheelnodes_scm_2k.npy"
# 500 nodes
#wheelnodesfile = "wheelnodes_scm_500.npy"


#numwnodes = 975
#numwnodes = 500
numwnodes = 375
#numwnodes = 325
#numwnodes = 195
#numwnodes = 125
wheelnodesfile = "wheelnodes_scm_"+str(numwnodes)+".npy"
print("Num tire nodes:", numwnodes)

# SCM
"""
#wheelnodestype = "full"
wheelnodestype = "fps"
#wheelnodestype = "sur"
if wheelnodestype=="full":
    wheelnodesfile = "wheelnodes_full.npy"
elif wheelnodestype=="fps":
    wheelnodesfile = "wheelnodes_new.npy"
elif wheelnodestype=="sur":
    wheelnodesfile = "wheelnodes.npy"
print("Using wheel nodes:",wheelnodestype)
"""

# SCM
# Sampling distance
sampledist = 0.46

#
#Not use sinkage, working yet!!!!!!!!!!
#
use_sinkage = True
#use_sinkage = False
print("Using sinkage:",use_sinkage)

use_vel = True
print("Using velocity:",use_vel)

#use_cosineloss = True
use_cosineloss = False

print("Using cosine loss:",use_cosineloss)

droprate = 0.0
print("Random drop rate:",droprate)

use_norm = None
#use_norm = "graph"
print("Using normalization layers:",use_norm)

sched_type = "CyclicLR"
#sched_type = "cos"
patience = 10000

use_amp = True
use_amp = False
print("Using AMP:",use_amp)

use_hmap = True
use_hmap = False
print("Using cnn hmap:",use_hmap)

# USE FALSE ALWAYS FOR NOW
use_wheelnodes = False
print("Using wheel nodes:",use_wheelnodes)

if use_wheelnodes:
    use_relposwheel = False
else:
    use_relposwheel = True
print("Using use rel pos wheel:",use_relposwheel)

# Fixed grid size for input height maps
sizegrid = 12

# For hmap
deltamap = 0.05
#def_min, def_max = -1.2, 0.

use_log = True
print("Using log",use_log)
normalize = True
print("Using normalization",normalize)

sinkmax = 0.5
linvelnorm = 10.
angvelnorm = 245.
globnorm = torch.tensor([linvelnorm,linvelnorm,linvelnorm,angvelnorm,angvelnorm,angvelnorm])

use_rollout = False
rol_len = 50
print("Using rollouts",use_rollout, ", Rollout lenght:",rol_len)

margin = 0.05