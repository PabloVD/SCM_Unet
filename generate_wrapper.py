

from Source.training import *
from Source.wrapper import *
#from Source.notoptimized_wrapper import *
from torch_geometric.loader import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity



def wheeldir_t(quat):

        orient2d = torch.zeros(2, dtype=torch.float32)
        quat = quat.view(4,1)


        yax = torch.cat([ (quat[1] * quat[2] - quat[0] * quat[3]) * 2, (quat[0] * quat[0] + quat[2] * quat[2]) * 2 - 1, (quat[2] * quat[3] + quat[0] * quat[1]) * 2])
        zax = torch.tensor([0,0,1], dtype=torch.float32)

        xax = torch.cross(yax, zax)

        orient2d = xax[:2]/torch.norm(xax[:2])

        return orient2d


#----------------
# Create wrapper
#----------------

input_channels = 3
global_emb_dim = 6

model = Unet(input_channels = input_channels,
             num_layers = n_layers,
             hidden_channels_in = hidden_channels,
             global_emb_dim = global_emb_dim)

deltastep = 1
n_sims = None
dataname = "DefSimsNoDamp"

namerun = "unet_"
namerun += dataname
namerun += "_test14"
namerun += "_deltastep_"+str(deltastep)
namerun += "_nsims_"+str(n_sims)
if use_log:
    namerun += "_log"
else:
    namerun += "_lin"
namerun += "_lrfact_{:.1e}".format(lr_fact)
namerun += "_gridsize_{:d}".format(sizegrid)
if use_rollout:
    namerun += "_rollout_"+str(rol_len)
else:
    namerun += "_singlestep"
namerun += "_margin_{:.1e}".format(margin)

namerun += "_lays_{:d}_std_{:.1e}_chan_{:d}_batch_{:d}".format(n_layers, noise_std, hidden_channels, batch_size)
namerun += "_inputs_{:d}_globdim_{:d}".format(input_channels, global_emb_dim)

if use_wheeltype:
    namerun += "_wheeltype"

sufix = "_"+namerun+"_lrs_{:.1e}_{:.1e}".format(lr_min, lr_max)

bestmodelname = path+"models/bestmodel"+sufix
#bestmodelname = path+"models/lastmodel"+sufix
state_dict = torch.load(bestmodelname, map_location=device)
model.load_state_dict(state_dict)
print(bestmodelname)
#"""
model.to(device)

#exit()

model.eval()

wrapmodel = Wrapper(model, namerun)

script_wrapper = torch.jit.script(wrapmodel)

sufixx = "_"+str(device)
namewrapper = "wrapped_unet"+sufixx+".pt"
script_wrapper.save(namewrapper)
chronobuildpath = "/home/tda/CARLA/chrono_scm_newcode/build_cuda/data/vehicle/terrain/scm/"
script_wrapper.save(chronobuildpath+namewrapper)


# Save in Unreal folder
#script_wrapper.save("/home/tda/CARLA/LastUnrealCARLA/"+"wrapped_gnn"+sufixx+".pt")

#----------------
# Prepare data
#----------------

n_sims = 1

#pathchrono = "/home/tda/Descargas/SCM_simulations/OverfitSim"
pathchrono = "/home/tda/Descargas/SCM_simulations/FlatSettling/Valid"

train_dataset = load_chrono_dataset(pathsims=pathchrono, numsims=n_sims, maxtimesteps = 250)

print("Sample graph:",train_dataset[0])


numwheels = 4

# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=numwheels, shuffle=False, num_workers=12)


step = -20
data = next(iter(train_loader))
#data.to(device)


#----------------
# Test wrapper in data
#----------------


pos = data.x[:,:,step]
part_types = data.part_types
condrig = (part_types==1)
pos_soil = pos[~condrig]
tirenodes = pos[condrig]
glob = data.glob[:,:,step]
wheeltype = data.wheeltype
batch = data.batch
batchsoil =  batch[~condrig]
batchrig = batch[condrig]

#hmap_nodes = torch.zeros((13041, 3))
#hmap_nodes = torch.load("sink_hmap_nodes")

#wheelpos = scatter(tirenodes, batchrig, dim=0, reduce="mean")
wheelpos = data.wheelpos[:,:,step]
#orientquat = torch.tensor([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype=torch.float32)
orientquat = data.quatorientation[:,:,step]

wheelframe = torch.zeros((numwheels,2))
for i in range(numwheels):
    wheelframe[i] = wheeldir_t(orientquat[i])

print(wheelpos.shape, pos.shape)


#glob += 0.01*torch.randn(glob.shape)
#glob[:,3] -= 1.
#print("Vels",glob.mean(0))
#print(data.force[:,:,step])

def get_pcloud_fixed(pcloud, wheelpos):

    xy_wheel = torch.round(wheelpos[:2]/deltamap)
    pcloud_grid = torch.round(pcloud[:,:2]/deltamap) - xy_wheel

    condbox_x = torch.logical_and( pcloud_grid[:,0]>-sizegrid//2-1, pcloud_grid[:,0]<sizegrid//2 )
    condbox_y = torch.logical_and( pcloud_grid[:,1]>-sizegrid//2-1, pcloud_grid[:,1]<sizegrid//2 )
    condbox = torch.logical_and( condbox_x, condbox_y )
    
    return pcloud[condbox]

def sampleinputdata(indbatch):

    condbatch = (batchsoil==indbatch)

    pos_soil_b = pos_soil[condbatch,:3]
    soil_vec = torch.cat([pos_soil_b,torch.zeros(pos_soil_b.shape[0],1)],dim=1) 

    soil_vec = get_pcloud_fixed(soil_vec, wheelpos[indbatch])

    #soil_vec = soil_vec[:sizegrid**2]

    wheeldata = [soil_vec, wheelpos[indbatch], orientquat[indbatch], glob[indbatch,:3], glob[indbatch,3:6]]

    return wheeldata

#print(pos.shape, glob.shape, wheeltype.shape, wheelpos.shape)


#"""
w_0 = sampleinputdata(0)
w_1 = sampleinputdata(1)
w_2 = sampleinputdata(2)
w_3 = sampleinputdata(3)


#vehicle_info = torch.zeros(3,dtype=torch.float32)#

loaded_wrapper = torch.jit.load(namewrapper)
#loaded_wrapper = wrapmodel
loaded_wrapper.clampder = True
print("Clamp diff force:",loaded_wrapper.clampder)

"""
hmap = loaded_wrapper.get_hmap(hmap_nodes)
hmap = hmap.view(1,1,hmap.shape[0],hmap.shape[1])
print("hmap",hmap_nodes.shape, hmap.shape)
hmap = hmap.repeat(4,1,1,1)
print("hmap",hmap_nodes.shape, hmap.shape)
exit()
"""

#prevstepsforce = data.force[:,:,step-prev_steps:step]
#loaded_wrapper.prevstepsforce = prevstepsforce.clone()
#print(torch.norm(loaded_wrapper.prevstepsforce[:,:3,-1],dim=1))
#print("Ground truth Fz",torch.norm(data.force[:,:3,step],dim=1))
#print("\nGround truth Fz",data.force[:,2,step])

print("\nLoaded wrapper")
#print("Using overlap condition:", loaded_wrapper.overcondition)

# For correctly clamping force
"""
prevprev_force = data.force[:,:,step-2]
prev_force = data.force[:,:,step-1]
loaded_wrapper.prevprev_force = prevprev_force.clone()
loaded_wrapper.prev_force = prev_force.clone()
"""
#print("Prev force",loaded_wrapper.prev_force)

#"""
for _ in range(20):
     out = loaded_wrapper(w_0, w_1, w_2, w_3, verbose=True)
#"""
     


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  

    out = loaded_wrapper(w_0, w_1, w_2, w_3, verbose=True)

    end.record()
    torch.cuda.synchronize()
    time_infer = start.elapsed_time(end)

print("Elapsed:",time_infer)

prof.export_chrome_trace("trace.json")

#print(out[0].shape)

exit()

#outwrapper = torch.cat([out[4].view(1,-1), out[5].view(1,-1), out[6].view(1,-1), out[7].view(1,-1)],dim=0)

"""
for i in range(4,8):
    print("Out",out[i][2])
#"""

"""
for i in range(0,4):
    soildef_w = out[i][:,2]
    soildefnonull = soildef_w[soildef_w!=0]
    #print("Out",soildefnonull[:10])
#"""

#print("Outsoil",out[0][:100,2])
#print(out[0].shape, w_0[0][:,2].shape)
#print("Data z",w_0[0][:5])
defz = out[0][:,2] - w_0[0][:,2].to(device)
print("Outsoil (wrapper)",defz[:10])
nonzeroindexes = torch.nonzero(defz)
print("Outsoil (wrapper, nonzero)",defz[nonzeroindexes][:10])

# soilpos = pos[~condrig]
# print(soilpos[:,0].min(),soilpos[:,0].max())
# print(out[0][:,0].min(),out[0][:,0].max())
# print(out[1][:,0].min(),out[1][:,0].max())
# print(out[2][:,0].min(),out[2][:,0].max())
# print(out[3][:,0].min(),out[3][:,0].max())
# #print(pos[~condrig][:5])
# exit()

#----------------
# Run model (without wrapper) in same data
#----------------

print("\nModel (no wrapper)")

#dataseq = torch.cat([pos, vel],dim=1)
data.soiltype = torch.zeros_like(data.soiltype)



dataseq = pos

dataseq, wheelpos, glob = to_wheelframe(wheelframe, dataseq, wheelpos, glob, rot90, batch)

#if model.use_sinkage:
sinkage = torch.zeros(dataseq.shape[0],1)
dataseq = torch.cat([dataseq, sinkage],dim=1)

if model.use_vel:
    vel = tire_vel(wheelpos, dataseq[:,:3], glob[:,:3], glob[:,3:6], data.batch)/vel_std
    dataseq = torch.cat([dataseq, vel],dim=1)

relpos = dataseq[:,:3] - wheelpos[batch]

if use_relposwheel:
    dataseq = torch.cat([dataseq,relpos],dim=-1)
window = sampleparts_rectangle(relpos)
dataseq_w = dataseq[window]
part_types_w = part_types[window]
batch_w = batch[window]

if use_hmap:
    hmap = loaded_wrapper.get_hmap(hmap_nodes)
    hmap = hmap.view(1,1,hmap.shape[0],hmap.shape[1])
    hmap = hmap.repeat(4,1,1,1)
    #print("hmap",hmap.shape)
    model.embed_hmap(hmap)

#prevprev_force = data.force[:,:,step-2]
#prev_force = data.force[:,:,step-1]
#print("Prev force",prev_force)
#print(loaded_wrapper.prev_force)

#glob = glob.to(device)
# use previous forces
"""
if model.prev_steps>0:
    glob = torch.cat([glob, norm_force(prev_force[:,:3]), norm_torque(prev_force[:,3:])],dim=1)
    if model.prev_steps>1:
        glob = torch.cat([glob, norm_force(prevprev_force[:,:3]), norm_torque(prevprev_force[:,3:])],dim=1)
"""

# use previous forces
# if prev_steps>0:
#     for ii in range(1,prev_steps+1):
#         forcestep_prev = prevstepsforce[:,:,-ii]
#         normforcestep_prev = norm_force(forcestep_prev[:,:3])
#         normtorqstep_prev = norm_torque(forcestep_prev[:,3:])
#         glob = torch.cat([glob, normforcestep_prev, normtorqstep_prev],dim=1)


#print("glob",glob[-1])
outsoil_w = model(dataseq_w.to(device), batch_w.to(device), part_types_w.to(device), glob.to(device), wheeltype.to(device), data.soiltype.to(device), wheelframe.to(device))

#outrig[:,:3] = unnorm_force(outrig[:,:3])
#outrig[:,3:6] = unnorm_torque(outrig[:,3:6])

"""
for i in range(4):
    print("Out",outrig[i][2])
#"""

outsoil = torch.zeros((dataseq[~condrig].shape[0],1), device=device)
outsoil[window[~condrig]] = outsoil_w

"""
for i in range(0,4):
    soildef_w = outsoil[i==batchsoil,0]
    soildefnonull = soildef_w[soildef_w!=0]
    #print("Out",soildefnonull[:10])
#"""



#print("Outsoil",outsoil[:100,0])

nonzeroindexes = torch.nonzero(outsoil[:,0])
print("Outsoil (no wrapper)",outsoil[:10])
print("Outsoil (no wrapper, nonzero)",outsoil[nonzeroindexes,0][:10])



"""
print("Real data")
for i in range(4):
    print("Real",data.force[i,2,step])
#"""