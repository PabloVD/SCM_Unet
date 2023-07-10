from Source.init import *
from Source.unet import *
#from Source.network_globnormglobframe import *
from typing import Tuple, List
from torch import Tensor
from typing import Optional, Tuple
import torch.profiler as profiler
import contextlib


WheelData = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

"""
```yml
INPUTS
# Inputs of the network are four tuples of tensors, and a height map, w_0, w_1, w_2, w_3
# w_i (for i 0 to 3) correspond to wheel related data and surrounding terrain
# Index corresponds to the order of the wheel: 0: left front, 1: right front, 2: left back, 3: right back
# Each of the w_i tuples of tensors has 5 tensors (with different shapes) at the current frame:
    - soil_vec: positions of soil particles and sinkage, x,y,z,s with shape (num_particles, 4)
    - wheel_pos: wheel position (3 values)
    - wheel_orient: wheel orientation quaternion (4 values)
    - wheel_lin_vel: linear velocity of the wheel (3 values)
    - wheel_ang_vel: angular velocity of the wheel (3 values)
# verbose: boolean to show debugging info

OUTPUTS
# Outputs of the network is a tuple of 4 tensors, the 3D deformation of each soil particle, for each of the 4 wheels, where
    - The first 2 axes are the x,y coordinates of the soil points
    - the 3rd axis is the z deformation

An example of a forward call of the wrapper:
for each i: w_i = [soil_vec, wheel_pos, wheel_orient, wheel_lin_vel, wheel_ang_vel]
out = wrapper(w_0, w_1, w_2, w_3, verbose)
soil0, soil1, soil2, soil3 = out

Note that num_particles is different for each wheel.
```
"""






class Wrapper(torch.nn.Module):

    def __init__(self, model, namemodel):
        super().__init__()

        self.model = model.eval()
        self.namemodel = namemodel
        self.device = torch.device(device)
        self.batchsize = 4
        self.use_float16 = False

        if self.use_float16:
            self.model = self.model.to(torch.float16)

        self.wheel_radius = 0.330229
        self.wheel_semiwidth = 0.2121/2.

        self.margin = margin

        self.window_x = self.wheel_radius*1.1
        self.window_y = self.wheel_semiwidth*1.1

        self.rot90 = torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float32,device=self.device)

        self.verbose = False

        print("Model employed:",self.namemodel)
        
        self.sizegrid = sizegrid
        self.deltamap = deltamap

        self.n_channels = self.model.input_channels

        self.xynodes = torch.zeros((self.batchsize*self.sizegrid**2,3),device=self.device)

        if "_log" in self.namemodel:
            self.use_log = True
        else:
            self.use_log = False

        self.normalize = normalize

        self.sinkmax = torch.tensor(sinkmax,device=self.device)
        self.globnorm = torch.tensor([linvelnorm,linvelnorm,linvelnorm,angvelnorm,angvelnorm,angvelnorm],device=self.device)

        self.soilpos = torch.zeros((self.batchsize*self.sizegrid**2, 4), device=self.device)
        self.hmap = torch.zeros((self.batchsize, self.n_channels, self.sizegrid, self.sizegrid), device=self.device)
        self.glob = torch.zeros((self.batchsize,6), device=self.device)
        self.wheelpos = torch.zeros((self.batchsize,3), device=self.device)
        self.quat = torch.zeros((self.batchsize,4), device=self.device)
        self.worient2d = torch.zeros((self.batchsize,2), device=self.device)

        self.zax = torch.tensor([0,0,1], dtype=torch.float32, device=self.device).repeat(self.batchsize,1)
        self.batch = torch.tensor([w for w in range(self.batchsize)], dtype=torch.int64, device=self.device).repeat_interleave(self.sizegrid**2)
        self.relpos = torch.zeros((self.batchsize*self.sizegrid**2,3),device=self.device)


    def dotprod2d(self, vec1, vec2):
    
        return torch.bmm(vec1.view(-1,1,2),vec2.view(-1,2,1)).squeeze()

    # decompose vector in components parallel, orthogonal to the wheel orientation, and vertical
    def vec_orientframe(self, diff, orient2d, normal2d):

        diff2d = diff[:,:2]
        diff_z = diff[:,2:3]
        diff_par = self.dotprod2d(diff2d,orient2d)
        diff_ort = self.dotprod2d(diff2d,normal2d)
        vecframe = torch.cat([diff_par.view(-1,1), diff_ort.view(-1,1), diff_z], dim=1)

        return vecframe

    def get_ort2d(self, wheelframe, rot90):
        
        numbatches = wheelframe.shape[0]
        norm2d = torch.bmm( rot90.repeat(numbatches,1,1), wheelframe.view(numbatches,2,1) ).squeeze(2)

        return norm2d

    # Move to wheel reference frame
    def to_wheelframe(self, wheelframe_step, x_step, batch):

        wheelframe_ort = self.get_ort2d(wheelframe_step, self.rot90)

        x_step = self.vec_orientframe(x_step, wheelframe_step[batch], wheelframe_ort[batch])

        return x_step

    
    def wheeldir(self, quat):

        quat = quat.view(self.batchsize,4,1)

        yax = torch.cat([ (quat[:,1] * quat[:,2] - quat[:,0] * quat[:,3]) * 2, (quat[:,0] * quat[:,0] + quat[:,2] * quat[:,2]) * 2 - 1, (quat[:,2] * quat[:,3] + quat[:,0] * quat[:,1]) * 2], dim=1)
        #zax = torch.tensor([0,0,1], dtype=torch.float32, device=self.device).repeat(self.batchsize,1)
        zax = self.zax
        xax = torch.cross(yax, zax)

        orient2d = xax[:,:2]/torch.norm(xax[:,:2],dim=-1,keepdim=True)

        return orient2d

    
    # relpos must be rotated wrt wheel
    def sampleparts_rectangle(self, relpos):

        cond_x = (torch.abs(relpos[:,0]) - self.window_x <= 0)
        cond_y = (torch.abs(relpos[:,1]) - self.window_y <= 0)
        #cond_z = (relpos[:,2] < 0)

        #condition = torch.logical_and(torch.logical_and(cond_x, cond_y),cond_z)
        condition = torch.logical_and(cond_x, cond_y)
        
        return condition
    
    
    def get_batched_hmap_wrapper(self, pcloud: Tensor, wheelpos: Tensor):

        sink = pcloud[:,3]
        pos = pcloud[:,:3]

        self.xynodes[:,:2] = pos[:,:2]

        self.relpos = pos - wheelpos[self.batch]

        rel_height = torch.clamp(-self.relpos[:,2], min=0., max=self.wheel_radius+self.margin)
        rel_dist = torch.norm(self.relpos,dim=1)
        rel_dist = torch.clamp(rel_dist, min=0., max=self.wheel_radius+self.margin)

        if self.normalize:
            rel_height = rel_height/self.wheel_radius
            rel_dist = rel_dist/self.wheel_radius
            sink = sink/self.sinkmax

        hmap_mat = torch.cat([rel_height.view(self.batchsize, 1, self.sizegrid, self.sizegrid), rel_dist.view(self.batchsize, 1, self.sizegrid, self.sizegrid), sink.view(self.batchsize, 1, self.sizegrid, self.sizegrid)],dim=1)
            
        return hmap_mat
    
    
    
    # # From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
    # def hmap2pcloud_rot_prev(self, outsoil, wheelpos, worient2d):

    #     outputs = []

    #     self.xynodes[:,2] = outsoil.view(-1)#self.batchsize*self.sizegrid**2)

    #     rot_pos = self.to_wheelframe(worient2d, self.relpos, self.batch)

    #     window = self.sampleparts_rectangle(rot_pos)

    #     outs = self.xynodes[window].to("cpu")

    #     redbatch = self.batch[window].to("cpu")

    #     for iw in range(self.batchsize):

    #         outputs.append( outs[iw==redbatch] )

    #     return outputs[0], outputs[1], outputs[2], outputs[3]
    
        
    # From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
    def hmap2pcloud_rot(self, outsoil, wheelpos, worient2d):

        outputs = []

        self.xynodes[:,2] = outsoil.view(-1)#self.batchsize*self.sizegrid**2)

        rot_pos = self.to_wheelframe(worient2d, self.relpos, self.batch)

        window = self.sampleparts_rectangle(rot_pos)

        outs = self.xynodes[window].to("cpu")

        redbatch = self.batch[window].to("cpu")

        for iw in range(self.batchsize):

            outputs.append( outs[iw==redbatch] )

        return outputs[0], outputs[1], outputs[2], outputs[3]
    
    
    

    def forward(self, soilpos: Tensor, wheelpos: Tensor, quat: Tensor, glob: Tensor):

        # Preprocess data

        #with profiler.record_function("Pre glob"):

        self.soilpos, self.wheelpos, self.quat, self.glob = soilpos.to(self.device), wheelpos.to(self.device), quat.to(self.device), glob.to(self.device)
        self.worient2d = self.wheeldir(self.quat)
        self.glob = self.glob/self.globnorm

        # soilpos = torch.cat([w[0] for w in wheels_info],dim=0).to(self.device)
        # self.wheelpos = torch.cat([w[1].view(1,3) for w in wheels_info],dim=0).to(self.device)
        # quat = torch.cat([w[2].view(1,4) for w in wheels_info],dim=0).to(self.device)
        # self.worient2d = self.wheeldir(quat)
        # linvel = torch.cat([w[3].view(1,3) for w in wheels_info],dim=0)
        # angvel = torch.cat([w[4].view(1,3) for w in wheels_info],dim=0)
        # self.glob = torch.cat([linvel, angvel],dim=1).to(self.device)/self.globnorm

        #with profiler.record_function("Get hmap"):

        self.hmap = self.get_batched_hmap_wrapper(self.soilpos, self.wheelpos)

        # torch.save(self.hmap,"hmap_new.pt")

        #print("hey")
        #print(self.hmap[2,0,:10,:10])
        #print(self.glob)

        if self.use_float16:
            self.hmap = self.hmap.to(torch.float16)
            self.glob = self.glob.to(torch.float16)

        # Feed forward NN step

        #with profiler.record_function("Forward"):

        # torch.save(self.hmap,"hmap_new.pt")
        # torch.save(self.glob,"glob_new.pt")

        outsoil = self.model(self.hmap, self.glob)

        #print(outsoil[2,0,:8,:8])
        #print(self.namemodel)
        #torch.save(outsoil,"outsoil_new.pt")

        if self.use_log:
            outsoil = -torch.exp(outsoil)

        if self.use_float16:
            outsoil = outsoil.to(torch.float32)

        # Postprocess data

        #with profiler.record_function("Post"):

        soil0, soil1, soil2, soil3 = self.hmap2pcloud_rot(outsoil, self.wheelpos, self.worient2d)

        return soil0, soil1, soil2, soil3
