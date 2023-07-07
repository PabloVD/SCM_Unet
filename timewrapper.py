

from Source.training import *
from Source.wrapper import *
#from Source.notoptimized_wrapper import *
from torch_geometric.loader import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt


def get_pcloud_fixed(pcloud, wheelpos):

    xy_wheel = torch.round(wheelpos[:2]/deltamap)
    pcloud_grid = torch.round(pcloud[:,:2]/deltamap) - xy_wheel

    condbox_x = torch.logical_and( pcloud_grid[:,0]>-sizegrid//2-1, pcloud_grid[:,0]<sizegrid//2 )
    condbox_y = torch.logical_and( pcloud_grid[:,1]>-sizegrid//2-1, pcloud_grid[:,1]<sizegrid//2 )
    condbox = torch.logical_and( condbox_x, condbox_y )
    
    return pcloud[condbox]

def sampleinputdata(indbatch, pos_soil_b, wheelpos, orientquat, glob):

    soil_vec = torch.cat([pos_soil_b,torch.zeros(pos_soil_b.shape[0],1)],dim=1) 

    soil_vec = get_pcloud_fixed(soil_vec, wheelpos[indbatch])
    soil_vec = soil_vec[:sizegrid**2]

    wheeldata = [soil_vec, wheelpos[indbatch], orientquat[indbatch], glob[indbatch,:3], glob[indbatch,3:6]]

    return wheeldata

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

loaded_wrapper = torch.jit.load(namewrapper)

# Save in Unreal folder
#script_wrapper.save("/home/tda/CARLA/LastUnrealCARLA/"+"wrapped_gnn"+sufixx+".pt")

#----------------
# Prepare data
#----------------

n_sims = 3
maxtimesteps = 2500

#pathchrono = "/home/tda/Descargas/SCM_simulations/OverfitSim"
#pathchrono = "/home/tda/Descargas/SCM_simulations/FlatSettling/Valid"

simspath = "/home/tda/Descargas/SCM_simulations/"
dataname = "DefSimsNoDamp"
pathvalid = simspath + dataname + "/5sims"

train_dataset = load_chrono_dataset(pathsims=pathvalid, numsims=n_sims, maxtimesteps = maxtimesteps)

print("Sample graph:",train_dataset[0])


numwheels = 12

# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=numwheels, shuffle=False, num_workers=12, drop_last=True)


#step = -20
#data = next(iter(train_loader))
#data.to(device)

#----------------
# Test wrapper in data
#----------------


time_tot = []

print("Len loader:", len(train_loader))



for data in train_loader:
    pbar = tqdm(range(maxtimesteps), total=maxtimesteps, position=0, leave=True, desc=f"Running...")
    for step in pbar:

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

        wheelpos = data.wheelpos[:,:,step]
        #orientquat = torch.tensor([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype=torch.float32)
        orientquat = data.quatorientation[:,:,step]

        wheelframe = torch.zeros((numwheels,2))
            

        ws = []

        #"""
        for i in range(numwheels):
            wheelframe[i] = wheeldir_t(orientquat[i])
            ws.append(sampleinputdata(i, pos_soil, wheelpos, orientquat, glob))
        
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  

        #out = loaded_wrapper(w_0, w_1, w_2, w_3, verbose=True)
        out = loaded_wrapper(*ws, verbose=True)

        end.record()
        torch.cuda.synchronize()
        time_infer = start.elapsed_time(end)
        time_tot.append(time_infer)


burnphase = 50
time_tot = np.array(time_tot)
time_tot = time_tot[burnphase:]
bins = 100

plt.figure(figsize=(12,10))
plt.hist(time_tot, bins=bins )
#plt.plot(time_tot, linestyle=":", color="r" )
plt.title("Mean time: {:.1e} +- {:.1e} ms".format(time_tot.mean(), time_tot.std()))
print("Mean time: {:.1e} +- {:.1e} ms".format(time_tot.mean(), time_tot.std()))
#plt.yscale("log")
plt.xlabel("time [ms]")
plt.savefig("time_numwheels_"+str(numwheels)+".png")
