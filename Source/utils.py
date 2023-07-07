from Source.init import *
import psutil
import time, datetime
import torchvision.transforms as tf
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

process = psutil.Process()

#--- Functions ---#

dim = 3

wheelnodes = np.load(wheelnodesfile)
wheelnodes -= wheelnodes.mean(0)
numwheelnodes = wheelnodes.shape[0]
wheelnodes = torch.tensor(wheelnodes,dtype=torch.float32)

def load_sim(datasetname, firsttimestep, maxtimesteps):

    # Print the memory (in GB) being used now:
    #process = psutil.Process()
    #print("Memory being used (GB):",process.memory_info().rss/1.e9)
    
    print("Loading: ",datasetname)

    soiltype = torch.tensor(0, dtype=torch.long).view(1,1)

    # Wheel info
    wheelfile = datasetname+"/wheel.npy"
    wheels = np.load(wheelfile)
    wheels = torch.tensor(wheels,dtype=torch.float32)
    wheels = wheels[:,:,firsttimestep:firsttimestep+maxtimesteps]

    soilfile = datasetname+"/soil.npy"
    soil = np.load(soilfile)
    soil = torch.tensor(soil,dtype=torch.float32)
    soil = soil[:,:,firsttimestep:firsttimestep+maxtimesteps]

    fourwheels = []

    for w in range(4):

        wheel = wheels[w]

        wheel_pos = wheel[:3]
        orientation = wheel[3:7]

        # Wheel nodes
        if use_wheelnodes:
            pos = torch.zeros((numwheelnodes,3,maxtimesteps),dtype=torch.float32)
            part_types = torch.ones(numwheelnodes,dtype=int)

            for t in range(maxtimesteps):

                orient_t = orientation[:,t].view(1,4)

                RotM = rotation_matrix_from_quaternion(orient_t, "cpu").squeeze()

                pos[:,:,t] = torch.matmul(RotM,wheelnodes.T).T + wheel_pos[:,t]
            
            minwheelx, maxwheelx = torch.min(pos[:,0,:]), torch.max(pos[:,0,:])
            minwheely, maxwheely = torch.min(pos[:,1,:]), torch.max(pos[:,1,:])

        else:
            pos = torch.empty((0,3,maxtimesteps),dtype=torch.float32)
            part_types = torch.empty(0,dtype=int)

            minwheelx, maxwheelx = torch.min(wheel_pos[0]), torch.max(wheel_pos[0])
            minwheely, maxwheely = torch.min(wheel_pos[1]), torch.max(wheel_pos[1])


        soil_w = soil.clone()

        # Subsample
        
        soil_w = soil_w[soil_w[:,0,0]>minwheelx-sampledist]
        soil_w = soil_w[soil_w[:,0,0]<maxwheelx+sampledist]
        soil_w = soil_w[soil_w[:,1,0]>minwheely-sampledist]
        soil_w = soil_w[soil_w[:,1,0]<maxwheely+sampledist]

        pos = torch.cat([pos,soil_w],dim=0)
        
        # Particle types
        part_types = torch.cat([part_types,torch.zeros(soil_w.shape[0],dtype=int)])

        globfeat =  wheel[7:13].view(1,-1,maxtimesteps)

        wheeltype = torch.tensor(w,dtype=torch.long).view(1,1)

        force = wheel[13:16].view(1,-1,maxtimesteps)

        if use_torque:
            force = torch.cat([force, wheel[16:19].view(1,-1,maxtimesteps)],dim=1)

        # Add 2d orientation
        wheelorient2d = wheeldir(orientation)
        wheelframe = wheelorient2d.view(1,-1,maxtimesteps)

        def_true = pos[:,2:3,1:].clone() - pos[:,2:3,:-1].clone()

        #graph = Data(x=pos, part_types=part_types, glob=globfeat, wheeltype=wheeltype, force=force, soiltype=soiltype, wheelframe=wheelframe)
        graph = Data(x=pos, part_types=part_types, glob=globfeat, wheeltype=wheeltype, force=force, soiltype=soiltype, wheelframe=wheelframe, quatorientation=orientation.view(1,-1,maxtimesteps), wheelpos=wheel_pos.view(1,3,maxtimesteps), def_true=def_true)
        # use_hmap
        #graph.hmap = hmap
        #graph.numsoilparts = torch.tensor(hmap.shape[0],dtype=torch.int).view(1,1)

        # Randomly rotate for data augmentation
        """
        if data_aug:
            ang = 2.*np.pi*random.random()
            graph = rotated_data(graph, ang)
        """

        nameburst = datasetname.replace("/home/tda/Descargas/SCM_simulations/Processed/","")+"_w"+str(w)

        graph.nameburst = nameburst

        #print("Size tensors:", graph.x.shape, sizetensor(graph.x))#, sizetensor(graph.glob), sizetensor(graph.force))

        
        # keys = [a for a in dir(graph) if not a.startswith('_') and not callable(getattr(graph, a)) ]
        # print(keys)
        # for key in keys:
        #     try:
        #         print(key, graph[key])
        #     except:
        #         continue
            
        # exit()

        fourwheels.append( graph )

    return fourwheels


def load_chrono_dataset(pathsims=pathchrono, numsims=None, firsttimestep = 0, maxtimesteps = 128, data_aug=data_aug, split_steps=False, use_rollout=False):

    assert not ((use_rollout==True) and (split_steps==True))

    print("Loading simulations from:",pathsims)

    foldersims = [ f.path for f in os.scandir(pathsims) if f.is_dir() ]

    random.shuffle(foldersims)
    #foldersims = sorted(foldersims)

    if numsims is not None:
        foldersims = foldersims[:numsims]

    dataset = []

    for datasetname in foldersims:

        fourwheels = load_sim(datasetname, firsttimestep, maxtimesteps)

        if split_steps:
            fourwheels = create_training_dataset(fourwheels)
            #del fourwheels
            #fourwheels = fourwheels_pre
        if use_rollout:
            fourwheels = create_rollout_dataset(fourwheels, rol_len=rol_len)

        # num_part_mean = 0 
        # tot_size = 0
        # for graph in fourwheels:
        #     num_part_mean += graph.x.shape[0]
        #     tot_size += sizetensor(graph.x)
        # print("Size tensors:", num_part_mean/len(fourwheels), tot_size)
        
        dataset.extend(fourwheels)

        #print("Memory being used (GB):",process.memory_info().rss/1.e9)

    print("\nNum sims", len(foldersims))
    print("Total num bursts", len(dataset))

    return dataset

def create_rollout_dataset(dataset, rol_len=10):

    stepdataset = []

    mxtps = dataset[0].x.shape[2]
    delta_steps = mxtps//rol_len

    for data in dataset:

        hmap_init = data.x[:,2:3,0]

        for idelta in range(delta_steps-1):

            stps = range(idelta*rol_len,(idelta+1)*rol_len)

            x_step = data.x[:,:,stps]
            glob_step = data.glob[:,:,stps]
            wheelframe_step = data.wheelframe[:, :, stps]
            quatorientation_step = data.quatorientation[:, :, stps]
            force_step = data.force[:,:,stps]
            wheelpos = data.wheelpos[:,:,stps]
            def_true = data.def_true[:,:,stps]

            minwheelx, maxwheelx = torch.min(wheelpos[0,0]), torch.max(wheelpos[0,0])
            minwheely, maxwheely = torch.min(wheelpos[0,1]), torch.max(wheelpos[0,1])
        
            condx = torch.logical_and(x_step[:,0,0]>minwheelx-sampledist, x_step[:,0,0]<maxwheelx+sampledist)
            condy = torch.logical_and(x_step[:,1,0]>minwheely-sampledist, x_step[:,1,0]<maxwheely+sampledist)
            condtot = torch.logical_and(condx, condy)

            x_step = x_step[condtot]
            hmap_init_w = hmap_init[condtot]
            def_true = def_true[condtot]

            #def_true = data.x[condsoil,2:3,idelta:idelta+rol_len+1] - data.x[condsoil,2:3,idelta:idelta+rol_len]
        
            graph = Data(x=x_step, part_types=data.part_types, glob=glob_step, wheeltype=data.wheeltype, force=force_step, soiltype=data.soiltype, wheelframe=wheelframe_step, quatorientation=quatorientation_step, wheelpos=wheelpos, def_true=def_true)
            graph.nameburst = data.nameburst
            graph.hmap_init = hmap_init_w
            graph.numnodes = x_step.shape[0]#.view(1,1)
            
            stepdataset.append(graph)

    print("Total num bursts", len(stepdataset))

    return stepdataset

def create_training_dataset(dataset_in, filter=True):

    stepdataset = []

    for datain in dataset_in:

        data = datain.clone()

        part_types = data.part_types.clone()

        #condsoil = (data.part_types.clone()==0)
        condsoil = (part_types==0)

        hmap_init = data.x[:,2:3,0].clone()

        for step in range(data.x.shape[2]-1):

            x_step = data.x[:,:,step].clone() 
            #z_ini = data.x[:,2:3,0].clone()
            glob_step = data.glob[:,:,step]
            #wheelframe_step = data.wheelframe[:, :, step]
            wheelpos_step = data.wheelpos[:, :, step]
            #quatorientation_step = data.quatorientation[:, :, step]
            #force_step = data.force[:,:,step]

            def_true = data.x[condsoil,2:3,step+1].clone() - data.x[condsoil,2:3,step].clone()

            #time_ini=time.time()

            relpos = x_step - wheelpos_step
            windowcond = sampleparts_rectangle(relpos)
            x_step = x_step[windowcond]
            def_true = def_true[windowcond[condsoil]]
            hmap_init_w = hmap_init[windowcond[condsoil]]
            #hmap_init_w = hmap_init

            #print("Elapsed window:",(time.time()-time_ini)*1.e3)
            #time_ini2 = time.time()

            in_hmap, def_hmap = get_hmap(x_step, wheelpos_step, hmap_init_w, def_true)

            #print("Elapsed get hmap:",(time.time()-time_ini2)*1.e3)
            #exit()

            # # Move to wheel reference frame
            # if use_equi:

            #     pseudobatch = torch.zeros(x_step.shape[0],dtype=int)
            #     x_step, wheelpos_step, glob_step = to_wheelframe(wheelframe_step, x_step, wheelpos_step, glob_step, rot90, pseudobatch)

            # if filter:

            #     relpos = x_step - wheelpos_step
            #     if use_equi:
            #         windowcond = sampleparts_rectangle(relpos)
            #     else:
            #         windowcond = sampleparts_cylinder(relpos)
            #     x_step = x_step[windowcond]
            #     def_true = def_true[windowcond[condsoil]]
            #     part_types_w = part_types.clone()[windowcond]
            #     z_ini_w = z_ini[windowcond]

            # else:

            #     part_types_w = part_types
            #     z_ini_w = z_ini
            if normalize:
                glob_step = glob_step/globnorm

            graph = Data(in_hmap=in_hmap, def_true=def_hmap, glob=glob_step, wheeltype=data.wheeltype, soiltype=data.soiltype)
            graph.nameburst = data.nameburst
            #graph.hmap_init = hmap_init

            # use_hmap
            #graph.hmap = data.hmap[:,:,step]
            #graph.numsoilparts = data.numsoilparts
            #graph.hmap = data.hmap[:,step:step+1,:,:]

            if in_hmap is not None:

                stepdataset.append(graph)

        del data, part_types

    #print("Memory being used (GB):",process.memory_info().rss/1.e9)
    del dataset_in
    #print("Memory being used (GB):",process.memory_info().rss/1.e9)
    

    return stepdataset


def sample_delta_dataset(dataset, deltastep = 1):

    stepdataset = []

    for data in dataset:

        x_step = data.x[:,:,::deltastep]
        glob_step = data.glob[:,:,::deltastep]
        wheelframe_step = data.wheelframe[:, :, ::deltastep]
        quatorientation_step = data.quatorientation[:, :, ::deltastep]
        force_step = data.force[:,:,::deltastep]

        graph = Data(x=x_step, part_types=data.part_types, glob=glob_step, wheeltype=data.wheeltype, force=force_step, soiltype=data.soiltype, wheelframe=wheelframe_step, quatorientation=quatorientation_step)
        graph.nameburst = data.nameburst
        # use_hmap
        #graph.hmap = data.hmap[:,:,::deltastep]
        #graph.hmap = data.hmap[:,::deltastep]
        #graph.numsoilparts = data.numsoilparts
        stepdataset.append(graph)

    return stepdataset



def get_hmap_old(pcloud, wheelpos, hmap_init, def_true, sizegrid=sizegrid):

    n_channels = 3
    
    npoints = pcloud.shape[0]
    #print(npoints)

    sink = hmap_init.view(-1) - pcloud[:,2]
    # Write coordinates relative to the wheel

    pcloud_rel = pcloud - wheelpos
    rel_dist = torch.norm(pcloud_rel,dim=1)
    rel_dist = torch.clamp(rel_dist, min=0., max=wheel_radius+deltamap)
    pcloud_rel[:,2] = torch.clamp(-pcloud_rel[:,2], min=0., max=wheel_radius*1.1)
    pcloud_grid = (pcloud_rel[:,:2]/deltamap).to(int)
    x_max, y_max = pcloud_grid[:,0].max(), pcloud_grid[:,1].max()
    
    pcloud_grid[:,0] += x_max
    pcloud_grid[:,1] += y_max
    xpoints = 2*x_max + 1
    ypoints = 2*y_max + 1

    hmap_mat = torch.zeros((n_channels, xpoints, ypoints))
    def_mat = torch.zeros((1, xpoints, ypoints))

    #print(pcloud_rel.shape, hmap_mat.shape, def_mat.shape)
    time_ini = time.time()

    for ic in range(npoints):
        cell = pcloud_grid[ic]
        ix, iy = cell[0].item(), cell[1].item()
        hmap_mat[0, ix, iy] = pcloud_rel[ic,2]
        hmap_mat[1, ix, iy] = rel_dist[ic]
        hmap_mat[2, ix, iy] = sink[ic]
        def_mat[0, ix, iy] = def_true[ic]

    #print("Elapsed grid loop:",(time.time()-time_ini)*1.e3)

    # Normalize
    #hmap_mat = (hmap_mat - def_min)/(def_max - def_min)
    hmap_mat = hmap_mat#/def_min

    hmap_mat = hmap_mat[:,xpoints//2 - sizegrid//2:xpoints//2 + sizegrid//2,ypoints//2 - sizegrid//2:ypoints//2 + sizegrid//2]
    def_mat = def_mat[:,xpoints//2 - sizegrid//2:xpoints//2 + sizegrid//2,ypoints//2 - sizegrid//2:ypoints//2 + sizegrid//2]

    #print(hmap_mat.shape, def_mat.shape)
        
    return hmap_mat.view(1,n_channels,sizegrid,sizegrid), def_mat.view(1,1,sizegrid,sizegrid)

def get_hmap(pcloud, wheelpos, hmap_init, def_true, sizegrid=sizegrid):

    n_channels = 3
    
    # Write coordinates relative to the wheel

    pcloud_rel = pcloud - wheelpos

    rel_height = torch.clamp(-pcloud_rel[:,2], min=0., max=wheel_radius+margin)
    rel_dist = torch.norm(pcloud_rel,dim=1)
    rel_dist = torch.clamp(rel_dist, min=0., max=wheel_radius+margin)
    sink = hmap_init.view(-1) - pcloud[:,2]

    xy_wheel = torch.round(wheelpos[:,:2]/deltamap)
    pcloud_grid = torch.round(pcloud[:,:2]/deltamap) - xy_wheel
 
    condbox_x = torch.logical_and( pcloud_grid[:,0]>-sizegrid//2-1, pcloud_grid[:,0]<sizegrid//2 )
    condbox_y = torch.logical_and( pcloud_grid[:,1]>-sizegrid//2-1, pcloud_grid[:,1]<sizegrid//2 )
    condbox = torch.logical_and( condbox_x, condbox_y )
    #print(pcloud_grid[condbox].shape)

    if pcloud_grid[condbox].shape[0]!=sizegrid*sizegrid:
        #print(pcloud_grid[condbox])
        return None, None

    rel_height = rel_height[condbox]
    rel_dist = rel_dist[condbox]
    sink = sink[condbox]
    def_true = def_true[condbox]
    #pcloud_window = pcloud[condbox] # only for validation

    if normalize:
        rel_height = rel_height/wheel_radius
        rel_dist = rel_dist/wheel_radius
        sink = sink/sinkmax

    hmap_mat = torch.cat([rel_height.view(1,sizegrid, sizegrid), rel_dist.view(1,sizegrid, sizegrid), sink.view(1,sizegrid, sizegrid)],dim=0)
    def_mat = def_true.view(1,sizegrid, sizegrid)

        
    return hmap_mat.view(1,n_channels,sizegrid,sizegrid), def_mat.view(1,1,sizegrid,sizegrid)#, pcloud_window


def get_hmap_batched(pcloud, wheelpos, hmap_init, def_true, batches, sizegrid=sizegrid):

    n_channels = 3
    
    # Write coordinates relative to the wheel

    #print(pcloud.shape, wheelpos.shape)

    pcloud_rel = pcloud - wheelpos

    rel_height = torch.clamp(-pcloud_rel[:,2], min=0., max=wheel_radius*1.1)
    rel_dist = torch.norm(pcloud_rel,dim=1)
    rel_dist = torch.clamp(rel_dist, min=0., max=wheel_radius+deltamap)
    sink = hmap_init.view(-1) - pcloud[:,2]

    xy_wheel = torch.round(wheelpos[:,:2]/deltamap)
    pcloud_grid = torch.round(pcloud[:,:2]/deltamap) - xy_wheel
 
    condbox_x = torch.logical_and( pcloud_grid[:,0]>-sizegrid//2-1, pcloud_grid[:,0]<sizegrid//2 )
    condbox_y = torch.logical_and( pcloud_grid[:,1]>-sizegrid//2-1, pcloud_grid[:,1]<sizegrid//2 )
    condbox = torch.logical_and( condbox_x, condbox_y )
    
    #print(pcloud_grid[condbox].shape, wheelpos.shape)

    if pcloud_grid[condbox].shape[0]!=sizegrid*sizegrid*batches:
        print("Not valid hmaps", pcloud_grid[condbox].shape, batches)
        return None, None, None

    rel_height = rel_height[condbox].view(batches,1,sizegrid, sizegrid)
    rel_dist = rel_dist[condbox].view(batches,1,sizegrid, sizegrid)
    sink = sink[condbox].view(batches,1,sizegrid,sizegrid)
    def_true = def_true[condbox].view(batches,1,sizegrid,sizegrid)
    pcloud_window = pcloud[condbox] # only for validation

    if normalize:
        rel_height = rel_height/wheel_radius
        rel_dist = rel_dist/wheel_radius
        sink = sink/sinkmax

    hmap_mat = torch.cat([rel_height, rel_dist, sink],dim=1)

    #print(hmap_mat.shape, def_true.shape)
        
    return hmap_mat, def_true, pcloud_window, condbox



def filter_deep_cases(dataset, margin=2.*linkradius):

    goodcases = []
    for data in dataset:
        x, parts = data.x, data.part_types
        if x.shape[0]> numwheelnodes:
            cond = (parts==indrig)
            xrig, xsoil = x[cond], x[~cond]
            if torch.min(xrig[:,2,:])-margin>torch.min(xsoil[:,2,:]):
                goodcases.append(data)
        elif x.shape[0]==numwheelnodes:
            goodcases.append(data)
    print("Filter cases by detph. Good bursts:",len(goodcases), "out of ",len(dataset))
    return goodcases

def add_car_orientation(datasetname, graph, burst, maxtimesteps):
    wheels = torch.zeros(4,3,maxtimesteps,dtype=torch.float32)
    for w in range(4):
        wheelpos = np.load(datasetname+"/wheel_{:d}_w{:d}.npy".format(burst,w))[:3]
        wheelpos = torch.tensor(wheelpos,dtype=torch.float32)
        wheels[w] = wheelpos
    orientunnorm = wheels[0,:2] + wheels[1,:2]-wheels[2,:2]-wheels[3,:2]
    orient = orientunnorm/torch.norm(orientunnorm,dim=0,keepdim=True)
    return torch.cat([graph.glob[:,:3], orient.view(1,2,maxtimesteps), graph.glob[:,3:]],dim=1)

# Separation between nodes ¡¡¡¡UPDATE!!!!
dx, dy, dz = 0., 0., 0.
seps = [dx, dz, dy]

def get_boundaries(x):
    step = 0
    pos = x[:,:,step]

    boundarylist = []
    for i in [0,1,2]:
        b_low, b_high = torch.min(pos[:,i])-seps[i], torch.max(pos[:,i])+seps[i]
        boundarylist.append([b_low, b_high])
    boundaries = torch.tensor(boundarylist, requires_grad=False, dtype=torch.float32)
    return boundaries

def wheeldir(quat_t):

    orient2d = torch.zeros(2,quat_t.shape[-1], dtype=torch.float32)

    for t in range(quat_t.shape[-1]):

        quat = quat_t[:,t]

        yax = torch.tensor(
        [ (quat[1] * quat[2] - quat[0] * quat[3]) * 2,
        (quat[0] * quat[0] + quat[2] * quat[2]) * 2 - 1,
        (quat[2] * quat[3] + quat[0] * quat[1]) * 2], dtype=torch.float32)
        zax = torch.tensor([0,0,1], dtype=torch.float32)

        xax = torch.cross(yax, zax)

        orient2d[:,t] = xax[:2]/torch.norm(xax[:2])

    return orient2d

def get_orientation(velrig, prevorient, batchrig):
    vel = global_mean_pool(velrig, batchrig)
    vel = vel[:,:2]
    normvel = torch.norm(vel)
    vel[normvel==0.]=prevorient
    vel[normvel!=0.]/=normvel
    return vel

def rot_vector(vec, ang, dimrot=3):

    if dimrot==3:
        rotmat = torch.tensor([[np.cos(ang), -np.sin(ang), 0.],
                                [np.sin(ang), np.cos(ang), 0.],
                                [0., 0., 1.]], dtype=torch.float32)
    elif dimrot==2:
        rotmat = torch.tensor([[np.cos(ang), -np.sin(ang)],
                                [np.sin(ang), np.cos(ang)]], dtype=torch.float32)

    #rotvec = torch.matmul(rotmat,vec.T).T
    #rotvec = torch.matmul(vec, rotmat.view(1,3,3).T).T.view(-1,3)
    #rotvec = torch.matmul(vec, rotmat.T.view(dimrot,dimrot,1)).T.view(-1,dimrot)
    rotvec = torch.matmul(vec, rotmat.T.view(dimrot,dimrot,1)).transpose(0,1).view(-1,dimrot)

    return rotvec

def rotated_data(data, ang):

    x = data.x.clone()
    glob = data.glob.clone()
    part_types = data.part_types.clone()
    #bounds = data.bounds.clone()
    wheeltype = data.wheeltype.clone()
    soiltype = data.soiltype.clone()
    force = data.force.clone()
    boundaryz = data.boundaryz.clone()

    #meanpoint = (bounds[0,:,0] + bounds[0,:,1])/2.
    #meanpoint[2] = 0.

    #meanpoint = torch.mean(x[~indrig,:,0], dim=0)
    meanpoint = torch.mean(data.x[:,:,0], dim=0)

    for t in range(x.shape[2]):

        x[:,:,t] = rot_vector(x[:,:,t]-meanpoint, ang) + meanpoint
        #newglob = torch.cat([glob[:,:,t],torch.zeros_like(glob[:,:1,t])],dim=1)
        #newglob = rot_vector(newglob, ang)
        #glob[:,:,t] = newglob[:,:2]
        glob[:,:3,t] = rot_vector(glob[:,:3,t], ang)    # linear velocity
        glob[:,3:5,t] = rot_vector(glob[:,3:5,t], ang, dimrot=2)    # horizontal orientation
        glob[:,5:8,t] = rot_vector(glob[:,5:8,t], ang)  # angular velocity
        force[:,:3,t] = rot_vector(force[:,:3,t], ang)    # force
        if use_torque:
            force[:,3:6,t] = rot_vector(force[:,3:6,t], ang)    # force

    #bounds = get_boundaries(x)
    #bounds = bounds.reshape(1,bounds.shape[0], bounds.shape[1])

    #new_data = Data(x=x, part_types=part_types, glob=glob, bounds=bounds, wheeltype=wheeltype)
    new_data = Data(x=x, part_types=part_types, glob=glob, wheeltype=wheeltype, soiltype=soiltype, force=force, boundaryz=boundaryz)

    return new_data




# From https://github.com/YunzhuLi/VGPL-Dynamics-Prior/blob/master/models.py and https://github.com/YunzhuLi/DPI-Net/blob/master/models.py
def rotation_matrix_from_quaternion(params, device):
    # params: (B * n_instance) x 4
    # w, x, y, z

    one = torch.ones(1, 1, device=device)
    zero = torch.zeros(1, 1, device=device)

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

def get_noise(position_sequence, noise_std):
    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    velocity_sequence = position_sequence[:,:,1:] - position_sequence[:,:,:-1]
    #num_velocities = velocity_sequence.shape[2]
    # Below, seq_len stands for the number of velocities
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape),device=device) * (noise_std/seq_len**0.5)

    # Apply the random walk.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=2)
    #print(velocity_sequence_noise.shape, secuencia[:,2:].shape)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler intergrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:,:,0:1]),
        torch.cumsum(velocity_sequence_noise, dim=2)], dim=2)

    return position_sequence_noise

# Write data at a timestep as a sequence including previous timesteps
def data_prev_steps(pos_seq, delta_t=dt):

    vel_seq = (pos_seq[:,:,1:] - pos_seq[:,:,:-1])/delta_t  # index 0 is vel in initial step, -1 is vel in last step
    vel_seq = vel_seq/vel_std
    vel_seq = vel_seq.reshape(pos_seq.shape[0],dim*seq_len)
    last_pos = pos_seq[:,:,-1]
    new_seq = torch.cat([last_pos,vel_seq],dim=1)

    return new_seq

def get_acceleration(prevstep, currstep, nextstep, delta_t=dt):

    acceleration = (nextstep -2*currstep + prevstep)/delta_t**2.
    acceleration = acceleration/acc_std

    return acceleration

# Verlet integrator, corrected for varying timestep
def pred_position(acceleration, currstep, prevstep, delta_t_0=dt, delta_t=dt):

    acceleration = acceleration*acc_std
    #most_recent_velocity = (currstep - prevstep)/delta_t
    #new_velocity = most_recent_velocity + acceleration*delta_t
    #new_position = currstep + new_velocity*delta_t
    new_position = currstep + (currstep - prevstep)*delta_t/delta_t_0 + acceleration*1./2.*delta_t*(delta_t + delta_t_0)

    return new_position

# Get memory of a tensor in Mb
def sizetensor(tensor):
    return tensor.element_size()*tensor.nelement()/1.e6

# Normal normalization
"""
def norm_force(force):
    return (force - force_mean)/force_std

def unnorm_force(force):
    return force*force_std + force_mean

def norm_torque(torque):
    return (torque - torque_mean)/torque_std

def unnorm_torque(torque):
    return torque*torque_std + torque_mean


# MinMax normalization
"""
def norm_force(force):
    #return (force - force_min)/(force_max - force_min)
    return (force - (force_min + force_max)/2.)/((force_max - force_min)/2.)

def unnorm_force(force):
    return force*((force_max - force_min)/2.) + (force_min + force_max)/2.

def norm_torque(torque):
    return (torque - (torque_min + torque_max)/2.)/((torque_max - torque_min)/2.)

def unnorm_torque(torque):
    return torque*((torque_max - torque_min)/2.) + (torque_min + torque_max)/2.
#"""

def tire_vel(wheelpos, wheelposparts, linvel, angvel, batch):

    #print(angvel[batch].shape, wheelposparts.shape,  wheelpos[batch].shape,  linvel[batch].shape)

    tirevel = torch.cross(angvel[batch], wheelposparts - wheelpos[batch], dim=1) + linvel[batch]

    #print(tirevel.shape)

    return tirevel

# relpos must be rotated wrt wheel
def sampleparts_rectangle(relpos):

    cond_x = (torch.abs(relpos[:,0]) - window_x < 0)
    cond_y = (torch.abs(relpos[:,1]) - window_y < 0)
    #cond_z = (relpos[:,2] < 0)

    #condition = torch.logical_and(torch.logical_and(cond_x, cond_y),cond_z)
    condition = torch.logical_and(cond_x, cond_y)
    
    return condition

# relpos not needed to be rotated wrt wheel
def sampleparts_cylinder(relpos):

    rad_hor = torch.sqrt(relpos[:,0]**2. + relpos[:,1]**2.)
    cond_hor = (rad_hor < window_radius)
    cond_vert = (relpos[:,2] < 0.)

    condition = torch.logical_and(cond_hor, cond_vert)
    
    return condition

# Change of reference frame methods

#rot90 = torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float32,device=device).view(1,2,2)
rot90 = torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float32).view(1,2,2)

def dotprod2d(vec1,vec2):
    
    return torch.bmm(vec1.view(-1,1,2),vec2.view(-1,2,1)).squeeze()

# decompose vector in components parallel, orthogonal to the wheel orientation, and vertical
def vec_orientframe(diff, orient2d, normal2d):

    diff2d = diff[:,:2]
    diff_z = diff[:,2:3]
    diff_par = dotprod2d(diff2d,orient2d)
    diff_ort = dotprod2d(diff2d,normal2d)
    vecframe = torch.cat([diff_par.view(-1,1), diff_ort.view(-1,1), diff_z], dim=1)

    return vecframe

def get_ort2d(wheelframe, rot90):
    
    numbatches = wheelframe.shape[0]
    norm2d = torch.bmm( rot90.repeat(numbatches,1,1), wheelframe.view(numbatches,2,1) ).squeeze(2)

    return norm2d

# def change_frame(vec3d, wheelframe, batch):
#     wheelframe_ort = get_ort2d(wheelframe, rot90)
#     vec3d = vec_orientframe(vec3d, wheelframe[batch], wheelframe_ort[batch])
#     return vec3d


# Move to wheel reference frame
def to_wheelframe(wheelframe_step, x_step, wheelpos_step, glob_step, rot90, batch):

    wheelframe_ort = get_ort2d(wheelframe_step, rot90)

    x_step = vec_orientframe(x_step, wheelframe_step[batch], wheelframe_ort[batch])
    #x_step = vec_orientframe(x_step, wheelframe_step.repeat(x_step.shape[0],1), wheelframe_ort.repeat(x_step.shape[0],1))

    wheelpos_step = vec_orientframe(wheelpos_step, wheelframe_step, wheelframe_ort)
    glob_step[:,:3] = vec_orientframe(glob_step[:,:3], wheelframe_step, wheelframe_ort)
    glob_step[:,3:6] = vec_orientframe(glob_step[:,3:6], wheelframe_step, wheelframe_ort)

    return x_step, wheelpos_step, glob_step

# Data augmentation transformations

class RigidRotation:
    def __init__(self):
        self.angles = [0,90,180,270]

    def __call__(self, x):
        randind = torch.randint(0,len(self.angles),size=(1,)).item()
        angle = self.angles[randind]
        return tf.functional.rotate(x, angle)

def get_transform():
    transforms = []
    transforms.append(RigidRotation())
    transforms.append(tf.RandomHorizontalFlip())
    return tf.Compose(transforms)

transform_dataaug = get_transform()

def dataaug_old(img, tar):
    state = torch.get_rng_state()
    img = transform_dataaug(img)
    torch.set_rng_state(state)
    tar = transform_dataaug(tar)
    return img, tar

angles = [0,90,180,270]
flips = [0,1]
rotmat_list = [ torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],dtype=torch.float32,device=device).view(1,2,2) for angle in angles]

def data_augmentation(img, tar, glob):

    # Rigid rotations
    randind = torch.randint(0,len(angles),size=(1,)).item()
    angle = angles[randind]
    rotmat = rotmat_list[randind].repeat(glob.shape[0],1,1)
    
    img = tf.functional.rotate(img, angle)
    tar = tf.functional.rotate(tar, angle)

    glob[:,:2] = torch.bmm(rotmat,glob[:,:2].unsqueeze(-1)).squeeze(-1)
    glob[:,3:5] = torch.bmm(rotmat,glob[:,3:5].unsqueeze(-1)).squeeze(-1)

    # Reflections
    hflip, vflip = torch.randint(0,len(flips),size=(1,)).item(), torch.randint(0,len(flips),size=(1,)).item()

    # Flip in x axis
    if hflip:
        img = tf.functional.hflip(img)
        tar = tf.functional.hflip(tar)
        glob[:,0] = -glob[:,0]
        glob[:,4] = -glob[:,4]  # Angular vel is pseudovector, modify y axis

    # Flip in y axis
    if vflip:
        img = tf.functional.vflip(img)
        tar = tf.functional.vflip(tar)
        glob[:,1] = -glob[:,1]
        glob[:,3] = -glob[:,3]  # Angular vel is pseudovector, modify x axis

    return img, tar, glob

# From hmap of shape (num_wheels, 1, sizegrid, sizegrid), returns a list of num_wheels point clouds, each with shape (sizegrid**2, 3), where the first two columns are the x,y global coordinates, and the third one is the vertical deformation
def hmap2pcloud(xynodes, hmap):

    #print(xynodes.shape, hmap.shape, wheelpos.shape, hmap.view(xynodes.shape[0],-1).shape)

    xynodes[:,2] += hmap.view(xynodes.shape[0])

    return xynodes.to("cpu")

def get_noise_hmap(hmap):

    shape = hmap.shape
    noise = noise_std*torch.randn(shape, device=device)
    # hmap += noise

    # hmap[:,0] = torch.clamp(hmap[:,0], min=0., max=wheel_radius+margin)    
    # hmap[:,1] = torch.clamp(hmap[:,1], min=0., max=wheel_radius+margin)

    #noise[:,0] = torch.clamp(noise[:,0], min=0., max=wheel_radius+margin)    
    #noise[:,1] = torch.clamp(noise[:,1], min=0., max=wheel_radius+margin)
    hmap += noise

    return hmap
