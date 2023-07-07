from Source.init import *
from Source.unet import *
#from Source.network_globnormglobframe import *
from typing import Tuple, List
from torch import Tensor
from typing import Optional, Tuple

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
        self.namemodel = namemodel#.replace("models/","")
        self.vel_std = 1.   # m/s
        self.acc_std = 1.   # m/s^2
        self.device = torch.device(device)
        self.batchsize = 6

        force_mean = [0., 0., 3.5e3]
        force_std = [1.e3, 1.e3, 2.e3]
        torque_mean = [0., 0., 0.]
        torque_std = [350., 350., 180.]
        force_min = [-1.e4, -1.8e4, -1.e3]
        force_max = [1.e4, 1.8e4, 2.e4]
        torque_min = [-4.7e3, -3.5e3, -1.2e3]
        torque_max = [4.7e3, 3.5e3, 1.2e3]
        
        self.force_mean, self.force_std = torch.tensor(force_mean, device=device, dtype=torch.float32).view(1,3), torch.tensor(force_std, device=device, dtype=torch.float32).view(1,3)
        self.torque_mean, self.torque_std = torch.tensor(torque_mean, device=device, dtype=torch.float32).view(1,3), torch.tensor(torque_std, device=device, dtype=torch.float32).view(1,3)
        self.force_min, self.force_max = torch.tensor(force_min, device=device, dtype=torch.float32).view(1,3), torch.tensor(force_max, device=device, dtype=torch.float32).view(1,3)
        self.torque_min, self.torque_max = torch.tensor(torque_min, device=device, dtype=torch.float32).view(1,3), torch.tensor(torque_max, device=device, dtype=torch.float32).view(1,3)


        #"""
        wheelnodes = np.load(wheelnodesfile)
        wheelnodes -= wheelnodes.mean(0)
        self.numwheelnodes = wheelnodes.shape[0]
        self.wheelnodes = torch.tensor(wheelnodes,dtype=torch.float32,device=self.device)
        self.batchtirenodes = torch.ones(self.numwheelnodes, dtype=torch.int64, device=self.device)
        #"""

        #self.wheel2dini = torch.tensor([1.,0.],dtype=torch.float32)
        """
        self.maxforcex = torch.tensor(5.e3,dtype=torch.float32,device=self.device)
        self.maxforcez = torch.tensor(1.5e4,dtype=torch.float32,device=self.device)
        self.maxtorquex = torch.tensor(1.e3,dtype=torch.float32,device=self.device)
        self.maxtorquez = torch.tensor(1.e3,dtype=torch.float32,device=self.device)
        self.maxforce = torch.cat([self.maxforcex.view(1), self.maxforcex.view(1), self.maxforcez.view(1)])
        self.maxtorque = torch.cat([self.maxtorquex.view(1), self.maxtorquex.view(1), self.maxtorquez.view(1)])
        """
        marginfactor = 1.1
        self.maxoutrig = marginfactor*torch.cat([self.force_max, self.torque_max],dim=0).view(-1)
        #self.maxoutrig *= 0.8
        self.minoutrig = marginfactor*torch.cat([self.force_min, self.torque_min],dim=0).view(-1)
        #self.minoutrig[2] = -200.

        self.maxoutsoil = 50.*torch.ones(3,dtype=torch.float32,device=self.device)
        self.minoutsoil = -self.maxoutsoil

        self.zeroout = torch.zeros(6,dtype=torch.float32,device=self.device)
        self.sinkfactor = 0.
        self.threshforcez = 5.e2

        self.wheel_radius = 0.330229
        self.wheel_semiwidth = 0.2121/2.
        # wheel_radius + 2.*r_link
        # wheel_width + 2.*r_link

        self.window_x = self.wheel_radius*1.1
        self.window_y = self.wheel_semiwidth*1.1
        #self.window_radius = self.wheel_radius*1.1
        self.margin = 0.03
        #self.margin = 0.1

        self.threshparts = 10

        self.rot90 = torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float32,device=self.device)

        #self.overcondition = True
        self.overcondition = False

        #self.local_frame = True
        self.local_frame = False
        self.wheel2d_local = torch.tensor([1.,0.],dtype=torch.float32, device=self.device) # Use if local frame in unreal is employed
        
        #self.verbose = True
        self.verbose = False

        self.count = 0

        #self.outdebug = True
        self.outdebug = False
        self.debugfolder = "/home/tda/CARLA/LastUnrealCARLA/debugouts"
        #self.debugfolder = "/home/tda/CARLA/FlatTerrainCARLA/debugouts"

        self.windowsample = True

        self.heightthreshold = 0.06

        # Minimum and maximum deformation, depends on step size
        self.def_min, self.def_max = -0.06, 0.001   # HiFreq Flat settling dataset
        #self.def_min, self.def_max = -0.09, 0.001   # deltastep=10 Flat settling dataset

        print("Model employed:",self.namemodel)

        # Maximum difference allowed in norm of force and torque between current and previous timestep
        #self.max_dif_force = torch.tensor([1.3e3, 800],dtype=torch.float32, device=self.device)    # for forward derivative
        #self.max_dif_force = torch.tensor([2.3e3, 400],dtype=torch.float32, device=self.device)     # for centered derivative
        self.max_dif_force = torch.tensor([2.e3, 400],dtype=torch.float32, device=self.device)     # try this

        """
        self.prev_force = torch.zeros((4,6),dtype=torch.float32, device=self.device)
        self.prevprev_force = torch.zeros((4,6),dtype=torch.float32, device=self.device)
        self.i_diff = 2
        """
        #self.prevstepsforce = torch.zeros((4,6,self.model.prev_steps),dtype=torch.float32, device=self.device)

        #self.clampder = False

        #self.use_wheelnodes = model.use_wheelnodes
        #self.use_relposwheel = model.use_relposwheel

        self.sizegrid = sizegrid
        self.nnodes = self.sizegrid**2
        self.deltamap = deltamap


        self.n_channels = self.model.input_channels

        self.pcloud_ini = self.ini_pcloud()
        self.pcloud_def = torch.zeros(self.nnodes,1,device=self.device)

        self.def_threshold = -1.e-6

        self.xynodes = torch.zeros((self.batchsize,self.sizegrid**2,3),device=self.device)

        if "_log" in self.namemodel:
            self.use_log = True
        else:
            self.use_log = False

        self.normalize = normalize

        self.sinkmax = torch.tensor(sinkmax,device=self.device)
        self.globnorm = torch.tensor([linvelnorm,linvelnorm,linvelnorm,angvelnorm,angvelnorm,angvelnorm],device=self.device)

        self.nodesmaxperwheel = 82

        self.pseudobatch = torch.zeros(self.nnodes,dtype=torch.int64,device=self.device)


        self.hmap = torch.zeros((self.batchsize, self.n_channels, self.sizegrid, self.sizegrid), device=self.device)
        self.glob = torch.zeros((self.batchsize,6), device=self.device)
        self.wheelpos = torch.zeros((self.batchsize,3), device=self.device)
        self.worient2d = torch.zeros((self.batchsize,2), device=self.device)


    def ini_pcloud(self):

        pcloud_ini = torch.zeros(self.nnodes,2,device=self.device)

        for i in range(self.sizegrid):
            for j in range(self.sizegrid):
                pcloud_ini[i*self.sizegrid + j,0] = (i-self.sizegrid//2)*self.deltamap
                pcloud_ini[i*self.sizegrid + j,1] = (j-self.sizegrid//2)*self.deltamap

        return pcloud_ini

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

        #wheelpos_step = self.vec_orientframe(wheelpos_step, wheelframe_step, wheelframe_ort)
        #glob_step[:,:3] = self.vec_orientframe(glob_step[:,:3], wheelframe_step, wheelframe_ort)
        #glob_step[:,3:6] = self.vec_orientframe(glob_step[:,3:6], wheelframe_step, wheelframe_ort)

        return x_step#, wheelpos_step, glob_step

    # From https://github.com/YunzhuLi/VGPL-Dynamics-Prior/blob/master/models.py and https://github.com/YunzhuLi/DPI-Net/blob/master/models.py
    def rotation_matrix_from_quaternion(self, params):
        # params: (B * n_instance) x 4
        # w, x, y, z

        one = torch.ones(1, 1, device=self.device)
        zero = torch.zeros(1, 1, device=self.device)
        #one = torch.ones(1, 1)
        #zero = torch.zeros(1, 1)

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params, dim=1, keepdim=True)
        w, x, y, z = \
                params[:, 0].view(-1, 1, 1), params[:, 1].view(-1, 1, 1), \
                params[:, 2].view(-1, 1, 1), params[:, 3].view(-1, 1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 2),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 2),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 2)), 1)

        # Above matrix is transpose of correct matrix, so correct it
        rot = rot.transpose(1, 2)

        # rot: (B * n_instance) x 3 x 3
        return rot

    def tire_pos(self, wheelpos, wheelorient):

        RotM = self.rotation_matrix_from_quaternion(wheelorient.view(1,4)).squeeze()
        tirepos = torch.matmul(RotM,self.wheelnodes.T).T + wheelpos

        return tirepos

    def tire_vel(self, wheelpos, wheelposparts, linvel, angvel):

        tirevel = torch.cross(angvel.view(1,-1), wheelposparts - wheelpos.view(1,-1), dim=1) + linvel.view(1,-1)

        return tirevel

    def wheeldir(self, quat):

        orient2d = torch.zeros(2, dtype=torch.float32, device=self.device)
        quat = quat.view(4,1)


        yax = torch.cat([ (quat[1] * quat[2] - quat[0] * quat[3]) * 2, (quat[0] * quat[0] + quat[2] * quat[2]) * 2 - 1, (quat[2] * quat[3] + quat[0] * quat[1]) * 2])
        zax = torch.tensor([0,0,1], dtype=torch.float32, device=self.device)

        xax = torch.cross(yax, zax)

        orient2d = xax[:2]/torch.norm(xax[:2])

        return orient2d

    def part_distr(self, relpos: Tensor, wheel: int, typepart: str):
        
        bins_x = torch.linspace(-3.*self.wheel_radius, 3.*self.wheel_radius, steps=50)
        bins_y = torch.linspace(-3.*self.wheel_width, 3.*self.wheel_width, steps=50)
        bins_z = torch.linspace(-2.*self.wheel_radius, 2.*self.wheel_radius, steps=50)
        binslist = [bins_x, bins_y, bins_z]
        for axx in [0,1,2]:
            disthist = torch.histogram(relpos[:,axx].cpu().detach(), binslist[axx], density=True)[0]
            torch.save(disthist,self.debugfolder+"/disthist_w"+str(wheel)+"_"+typepart+"_"+str(self.count)+"_"+str(axx))


    def vel_distr(self, vel: Tensor, wheel: int, typepart: str):
        
        bins_x = torch.linspace(-5, 5., steps=100)
        bins_y = torch.linspace(-5, 5., steps=100)
        bins_z = torch.linspace(-5, 5., steps=100)
        binslist = [bins_x, bins_y, bins_z]
        for axx in [0,1,2]:
            #if axx==2:
            #    relpos[:,axx]+=self.wheel_radius    # position around the bottom of the wheel
            disthist = torch.histogram(vel[:,axx].cpu().detach(), binslist[axx], density=True)[0]
            torch.save(disthist,self.debugfolder+"/velhist_w"+str(wheel)+"_"+typepart+"_"+str(self.count)+"_"+str(axx))

    # # relsoil must be rotated wrt wheel
    # def sampleparts(self, relpos):

    #     rad_hor = torch.sqrt(relpos[:,0]**2. + relpos[:,1]**2.)
    #     cond_hor = (rad_hor < self.window_radius)
    #     cond_vert = (relpos[:,2] < 0.)
    #     condition = torch.logical_and(cond_hor, cond_vert)
    #     #condition = cond_hor
        
    #     return condition
    
    # relpos must be rotated wrt wheel
    def sampleparts_rectangle(self, relpos):

        cond_x = (torch.abs(relpos[:,0]) - self.window_x <= 0)
        cond_y = (torch.abs(relpos[:,1]) - self.window_y <= 0)
        #cond_z = (relpos[:,2] < 0)

        #condition = torch.logical_and(torch.logical_and(cond_x, cond_y),cond_z)
        condition = torch.logical_and(cond_x, cond_y)
        
        return condition
    
        
    # def get_hmap_wrapper_old(self, pcloud: Tensor, wheelpos: Tensor, sink: Tensor, wheelind: int):
        
    #     npoints = pcloud.shape[0]

    #     # Write coordinates relative to the wheel

    #     pcloud_rel = pcloud - wheelpos
    #     rel_dist = torch.norm(pcloud_rel,dim=1)
    #     rel_dist = torch.clamp(rel_dist, min=0., max=self.wheel_radius+self.deltamap)
    #     pcloud_rel[:,2] = torch.clamp(-pcloud_rel[:,2], min=0., max=self.wheel_radius*1.1)
    #     pcloud_grid = (pcloud_rel[:,:2]/self.deltamap).to(torch.int64)
    #     x_max, y_max = pcloud_grid[:,0].max(), pcloud_grid[:,1].max()
        
    #     pcloud_grid[:,0] += x_max
    #     pcloud_grid[:,1] += y_max
    #     xpoints = 2*x_max + 1
    #     ypoints = 2*y_max + 1

    #     hmap_mat = torch.zeros((self.n_channels, int(xpoints), int(ypoints)),device=self.device)

    #     #print(pcloud_rel.shape, hmap_mat.shape, sink.shape)

    #     for ic in range(npoints):
    #         #print(ic)
    #         cell = pcloud_grid[ic]
    #         ix, iy = int(cell[0].item()), int(cell[1].item())
    #         hmap_mat[0, ix, iy] = pcloud_rel[ic,2]
    #         hmap_mat[1, ix, iy] = rel_dist[ic]
    #         hmap_mat[2, ix, iy] = sink[ic]

    #     # Normalize
    #     #hmap_mat = (hmap_mat - def_min)/(def_max - def_min)
    #     #hmap_mat = hmap_mat#/def_min

    #     #print("hmap shape pre",hmap_mat.shape)

    #     hmap_mat = hmap_mat[:,xpoints//2 - self.sizegrid//2:xpoints//2 + self.sizegrid//2,ypoints//2 - self.sizegrid//2:ypoints//2 + self.sizegrid//2]

    #     #print("hmap shape",hmap_mat.shape)
    #     #print(hmap_mat.shape, def_mat.shape)
            
    #     return hmap_mat
    

    # # Maybe use .to(int) rather than floor
    # def get_hmap_wrapper_prev(self, pcloud: Tensor, wheelpos: Tensor, sink: Tensor, wheelind: int):
        
    #     # Write coordinates relative to the wheel

    #     #print(pcloud.shape, sink.shape)

    #     pcloud_rel = pcloud - wheelpos.view(1,3)

    #     rel_height = torch.clamp(-pcloud_rel[:,2], min=0., max=self.wheel_radius*1.1)
    #     rel_dist = torch.norm(pcloud_rel,dim=1)
    #     rel_dist = torch.clamp(rel_dist, min=0., max=self.wheel_radius+self.deltamap)

    #     xy_wheel = torch.round(wheelpos[:2]/self.deltamap)
    #     pcloud_grid = torch.round(pcloud[:,:2]/self.deltamap) - xy_wheel
    
    #     condbox_x = torch.logical_and( pcloud_grid[:,0]>-self.sizegrid//2-1, pcloud_grid[:,0]<self.sizegrid//2 )
    #     condbox_y = torch.logical_and( pcloud_grid[:,1]>-self.sizegrid//2-1, pcloud_grid[:,1]<self.sizegrid//2 )
    #     condbox = torch.logical_and( condbox_x, condbox_y )

    #     #print(pcloud_grid[condbox].shape[0])

    #     if pcloud_grid[condbox].shape[0]!=self.sizegrid**2:
    #         #print(pcloud_grid[condbox])
    #         print("hmap not valid")
    #         return torch.zeros((self.n_channels, self.sizegrid, self.sizegrid),device=self.device)
        
    #     self.xynodes[wheelind,:,:2] = pcloud[condbox,:2]

    #     rel_height = rel_height[condbox]
    #     rel_dist = rel_dist[condbox]
    #     sink = sink[condbox]

    #     if self.normalize:
    #         rel_height = rel_height/self.wheel_radius
    #         rel_dist = rel_dist/self.wheel_radius
    #         sink = sink/self.sinkmax

    #     hmap_mat = torch.cat([rel_height.view(1,self.sizegrid, self.sizegrid), rel_dist.view(1,self.sizegrid, self.sizegrid), sink.view(1,self.sizegrid, self.sizegrid)],dim=0)
            
    #     return hmap_mat
    
    # Maybe use .to(int) rather than floor
    def get_hmap_wrapper(self, pcloud: Tensor, wheelpos: Tensor, sink: Tensor, wheelind: int):
        
        # Write coordinates relative to the wheel

        #print(pcloud.shape, sink.shape)

        self.xynodes[wheelind,:,:2] = pcloud[:,:2]

        pcloud_rel = pcloud - wheelpos.view(1,3)
        #if wheelind==0:
        #    print("Pcloud rel",pcloud, wheelpos)

        rel_height = torch.clamp(-pcloud_rel[:,2], min=0., max=self.wheel_radius*1.1)
        rel_dist = torch.norm(pcloud_rel,dim=1)
        rel_dist = torch.clamp(rel_dist, min=0., max=self.wheel_radius+self.deltamap)

        # xy_wheel = torch.round(wheelpos[:2]/self.deltamap)
        # pcloud_grid = torch.round(pcloud[:,:2]/self.deltamap) - xy_wheel
    
        # condbox_x = torch.logical_and( pcloud_grid[:,0]>-self.sizegrid//2-1, pcloud_grid[:,0]<self.sizegrid//2 )
        # condbox_y = torch.logical_and( pcloud_grid[:,1]>-self.sizegrid//2-1, pcloud_grid[:,1]<self.sizegrid//2 )
        # condbox = torch.logical_and( condbox_x, condbox_y )

        #print(pcloud_grid[condbox].shape[0])

        # if pcloud_grid.shape[0]!=self.sizegrid**2:
        #     #print(pcloud_grid[condbox])
        #     print("hmap not valid")
        #     return torch.zeros((self.n_channels, self.sizegrid, self.sizegrid),device=self.device)

        #print(self.xynodes[wheelind,:,:2])

        # rel_height = rel_height[condbox]
        # rel_dist = rel_dist[condbox]
        # sink = sink[condbox]

        if self.normalize:
            rel_height = rel_height/self.wheel_radius
            rel_dist = rel_dist/self.wheel_radius
            sink = sink/self.sinkmax

        hmap_mat = torch.cat([rel_height.view(1,self.sizegrid, self.sizegrid), rel_dist.view(1,self.sizegrid, self.sizegrid), sink.view(1,self.sizegrid, self.sizegrid)],dim=0)
            
        return hmap_mat
    
    # From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
    def hmap2pcloud_topk(self, hmap, wheelpos):

        #pclouds = []

        self.xynodes[:,:,2] = hmap.view(self.batchsize,-1)

        # DEBUGGG
        #self.xynodes = torch.zeros_like(self.xynodes)

        #print("we",self.xynodes[0,:5,:2], "wheelpos", wheelpos)
        #print("dist",self.xynodes.shape, wheelpos.shape)
        #dist = torch.abs(self.xynodes[:,:,:2] - wheelpos[:,:2].view(self.batchsize,1,2))
        #print("dist",dist.indices.topk(self.nodesmaxperwheel,largest=False).shape)

        outputs = []


        for iw in range(self.batchsize):
            #print(self.xynodes.shape)
            dist = torch.norm(self.xynodes[iw,:,:2] - wheelpos[iw,:2].view(1,2),dim=1)
            nearest = dist.topk(self.nodesmaxperwheel,largest=False).indices
            #print(dist.topk(self.nodesmaxperwheel,largest=False).values)
            outputs.append( self.xynodes[iw,nearest] )
    
        return outputs[0].to("cpu"), outputs[1].to("cpu"), outputs[2].to("cpu"), outputs[3].to("cpu")
    
    # From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
    def hmap2pcloud_rot(self, hmap, wheelpos, worient2d):


        self.xynodes[:,:,2] = hmap.view(self.batchsize,-1)

        outputs = []

        #print("pre",self.xynodes[0], wheelpos[0])
        #print(wheelpos)

        relpos = self.xynodes - wheelpos.view(self.batchsize,1,3)

        #print("rel",relpos[0])

        for iw in range(self.batchsize):

            rot_pos = self.to_wheelframe(worient2d[iw].view(1,-1), relpos[iw], self.pseudobatch)
            #rot_pos = self.to_wheelframe(worient2d[iw].view(1,-1), self.xynodes[iw] - wheelpos[iw].view(1,3), self.pseudobatch)
            #rot_pos = relpos[iw]

            #print("post",rot_pos[:10])

            window = self.sampleparts_rectangle(rot_pos)
            #print(window)
            #print(rot_pos.shape, self.xynodes[iw,window].shape)

            #dist = torch.norm(rot_pos[iw,:,:2],dim=1)
            #nearest = dist.topk(self.nodesmaxperwheel,largest=False).indices

            outputs.append( self.xynodes[iw,window] )

            #print( outputs[iw].shape )
    
        return outputs[0].to("cpu"), outputs[1].to("cpu"), outputs[2].to("cpu"), outputs[3].to("cpu")
    
    # From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
    # def hmap2pcloud_old(self, hmap, wheelpos):

    #     pclouds = []

    #     for w in range(self.batchsize):
    #         for i in range(self.sizegrid):
    #             for j in range(self.sizegrid):
    #                 self.pcloud_def[i*self.sizegrid + j,0] = hmap[w,0,i,j]

    #         pcloud = torch.cat([ self.pcloud_ini + wheelpos[w,:2].view(1,2), self.pcloud_def ], dim=1)
    #         #pcloud = torch.cat([ self.pcloud_ini, self.pcloud_def ], dim=1)
    #         pclouds.append(pcloud)

    #     return pclouds

    def getwheeldata(self, wheelind: int, wheel_list: WheelData):

        pos, wheelpos, wheelorient, linvel, angvel = wheel_list[0], wheel_list[1], wheel_list[2], wheel_list[3], wheel_list[4]
        pos, wheelpos, wheelorient, linvel, angvel = pos.to(self.device), wheelpos.to(self.device), wheelorient.to(self.device), linvel.to(self.device), angvel.to(self.device)
        
        sinkage = pos[:,3:4].view(-1)
        pos = pos[:,:3]

        #print(pos.shape, sinkage.shape, wheelpos.shape)

        #print(wheelind, pos[:10])

        hmap_w = self.get_hmap_wrapper(pos, wheelpos, sinkage, wheelind)
        # DEBUGGGG
        #hmap_w = torch.randn((self.n_channels,self.sizegrid,self.sizegrid),device=self.device)

        glob_w = torch.cat([linvel, angvel])#, steer.view(1), throt.view(1), brak.view(1)])

        if self.normalize:
            glob_w = glob_w/self.globnorm

        #print(hmap_w.shape, glob_w.shape)

        wheelorient2d = self.wheeldir(wheelorient)

        return hmap_w, glob_w, wheelpos, wheelorient2d

    def forward(self, w_0: WheelData, w_1: WheelData, w_2: WheelData, w_3: WheelData, w_4: WheelData, w_5: WheelData, verbose: bool = True):

        self.verbose = verbose

        #wheeltype = torch.zeros(self.batchsize, dtype=torch.int64, device=self.device)

        wheels_info = [w_0,w_1,w_2,w_3,w_4,w_5]

        #num_parts = [w_i[0].shape[0] for w_i in wheels_info]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        for wheelind in range(self.batchsize):
            #pos_w, vel_w, batch_w, part_types_w, wheelpos_w, glob_w, wheelorient2d_w, cond_w, distwheelfloor_w, sinkage_w = self.getwheeldata(wheelind, wheels_info[wheelind])
            hmap_w, glob_w, wheelpos_w, w2d_w = self.getwheeldata(wheelind, wheels_info[wheelind])
            self.hmap[wheelind] = hmap_w
            self.glob[wheelind] = glob_w
            self.wheelpos[wheelind] = wheelpos_w
            self.worient2d[wheelind] = w2d_w

        # end.record()
        # torch.cuda.synchronize()
        # time_infer = start.elapsed_time(end)
        # print("Elapsed load data:",time_infer)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        outsoil = self.model(self.hmap, self.glob)

        if self.use_log:
            outsoil = -torch.exp(outsoil)

        # end.record()
        # torch.cuda.synchronize()
        # time_infer = start.elapsed_time(end)
        # print("Elapsed model:",time_infer)

        # print("pre clamp",outsoil[2,0,:4,:4])
        # outsoil = torch.clamp(outsoil, min=self.def_threshold)
        # print("clamped",outsoil[2,0,:4,:4])

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        #soil0, soil1, soil2, soil3 = self.hmap2pcloud_topk(outsoil, self.wheelpos)
        soil0, soil1, soil2, soil3 = self.hmap2pcloud_rot(outsoil, self.wheelpos, self.worient2d)

        #print(soil0.shape)

        # end.record()
        # torch.cuda.synchronize()
        # time_infer = start.elapsed_time(end)
        # print("Elapsed hmap2pcloud:",time_infer)

        #print(outsoil.shape, soil0.shape, soil2.shape)

        #print(soil2)
        #print(wheelpos)
        #print(soil0.shape)

        return soil0, soil1, soil2, soil3
