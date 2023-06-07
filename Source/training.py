
#from Source.rigid_body import *
from Source.utils import *
from Source.visualize import *
from torch_scatter import scatter

rig_soil_fact = 1. #1.
sample_loss = False

if use_amp:
    # Automated Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

#--- TRAINING ---#

if use_cosineloss:
    criterion_r = torch.nn.L1Loss(reduction='sum')
    #criterion_r = torch.nn.SmoothL1Loss(reduction='sum')
else:
    criterion_r = torch.nn.MSELoss(reduction='sum')
criterion_s = torch.nn.MSELoss(reduction='sum')
#cosineemb = torch.nn.CosineEmbeddingLoss(reduction='sum')
#criterion = lambda x, y: cosineemb(x, y, 1)
cosinesim = torch.nn.CosineSimilarity()
#criterion = lambda x, y: torch.sum(1. - cosinesim(x, y),dim=0)



# Predict the following state at timestep step and compute loss
def singlestep(model, data, train=True, use_noise=True):

    #x, btch, part_types = data.x, data.batch, data.part_types

    hmap = data.in_hmap
    glob = data.glob
    def_true = data.def_true

    if train and data_aug:
        hmap, def_true = dataaug(hmap, def_true)

    outsoil = model(hmap, glob)

    loss_soil = criterion_s(outsoil, def_true)

    return loss_soil

   

# Predict sequence (rollout) starting at timestep in_step with a length rollout_seq and compute loss
def rollout(model, data, in_step=seq_len, rollout_seq=50):

    part_types = data.part_types.to(device)

    loss_rol = 0
    loss_rol_rig = 0

    maxtimesteps = data.x.shape[2]

    btch = data.batch.to(device)

    if in_step+rollout_seq>maxtimesteps-1:
        print("max step too large",in_step+rollout_seq-1)

    # Initialize the prediction tensor


    # SCM
    gnn_out = data.x.clone()    # Afterwards, this is updated!


    condrig = (part_types==1)
    #batchrig = btch[condrig]

    # For each time "step" and previous ones, predict step+1
    steps = range(in_step, in_step+rollout_seq)

    for step in steps:

        glob = data.glob[:,:,step].to(device)

        x_step = gnn_out[:,:,step].to(device)

        wheelpos_step = data.wheelpos[:,:,step].to(device)

        relpos = x_step - wheelpos_step
        windowcond = sampleparts_rectangle(relpos)
        x_step = x_step[windowcond]
        def_true = def_true[windowcond[condsoil]]
        hmap_init_w = hmap_init[windowcond[condsoil]]
        #hmap_init_w = hmap_init

        print("Elapsed window:",(time.time()-time_ini)*1.e3)
        time_ini2 = time.time()

        in_hmap, def_hmap = get_hmap(x_step, wheelpos_step, hmap_init_w, def_true)
        outsoil = model(hmap, glob)

        # Soil
        outsoil = torch.zeros((dataseq[~condrig].shape[0],1), device=device)
        outsoil[window[~condrig]] = outsoil_w

        # DEBUGGG
        #outsoil[window[~condrig]] = -0.01

        # SCM
        """
        pred_pos_soil = pred_position(outsoil, currsoil, prevsoil)
        #loss_rol += criterion(pred_pos_soil, nextsoil)

        if sample_loss:

            com_curr = global_mean_pool(currrig, batchrig)
            distparticles = torch.norm(currsoil - com_curr[data.batch[~condrig]],dim=1)
            closeparticles = (distparticles - 2.*wheel_radius <= 0.)

            soilloss = criterion(pred_pos_soil[closeparticles], nextsoil[closeparticles])#/nextsoil.shape[0]ç

        else:
            soilloss = criterion(pred_pos_soil, nextsoil)#/nextsoil.shape[0]

        """
        #def_true = nextsoil[:,2:3]-currsoil[:,2:3]
        z_pred = currsoil[:,2:3] + outsoil
        z_pred[~window[~condrig]] = nextsoil[~window[~condrig],2:3]

        #soilloss = criterion(outsoil, def_true)
        soilloss = criterion_s(z_pred[window[~condrig]], nextsoil[window[~condrig],2:3])
        #soilloss = torch.abs(outsoil - def_true).mean()

        

        #loss_rol += torch.log10(soilloss) + logloss_rig
        loss_rol += soilloss
        #loss_rol_rig += logloss_rig

       


        #gnn_out[window,2:3,step+1][~condrig] = z_pred[window[~condrig]]
        gnn_out[~condrig,2:3,step+1] = z_pred.cpu().detach()
        

        


    return loss_rol/len(steps), loss_rol_rig/len(steps), gnn_out, force_out



# Training routine
def train(model, loader, optimizer, scheduler, train_step, writer, lastmodelname):
    model.train()

    total_loss = 0
    pbar = tqdm(loader, total=len(loader), position=0, leave=True, desc=f"Training")
    for data in pbar:

        optimizer.zero_grad(set_to_none=True)

        data = data.to(device)

        # Train with single steps

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = singlestep(model, data)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = singlestep(model, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


        if scheduler is not None:
            scheduler.step()

        loss = loss.item()
        total_loss += loss
        if train_step % log_steps == 0:
            writer.add_scalar("Training loss per gradient step", loss, train_step)

        optimizer.zero_grad(set_to_none=True)  # Clear gradients.
        train_step+=1

        if (train_step+1)%steps_save_model==0:
            torch.save(model.state_dict(), lastmodelname)

    return total_loss / len(loader), train_step



# Evaluation routine
@torch.no_grad()
def test(model, loader, maxsteps, writer, vis=False):
    model.eval()

    total_loss = 0.
    tot_loss_rig = 0.
    for data in loader:
        #data = data.to(device)
        loss = 0

        loss_rol, loss_rig, gnn_out, force_out = rollout(model, data, in_step=seq_len, rollout_seq=maxsteps-seq_len-1)
        #print(gnn_out[:3,:,-3:])
        #print(force_out[:,:,-3:])
        total_loss += loss_rol.item()#/data.x.shape[0]
        #tot_loss_rig += loss_rig.item()#/data.x.shape[0]
        if vis:
            vis_results(data, gnn_out, in_step=0, lensteps=maxsteps-1, force_out=force_out, interval=100)

    #if writer is not None:
    #    compare_truth_pred(data, gnn_out, -1, writer, boundarylist)
        #compare_truth_pred(data, gnn_out, 60, writer, boundarylist)

    return total_loss / len(loader), tot_loss_rig / len(loader)

@torch.no_grad()
def test_singlesteps(model, loader):
    model.eval()

    total_loss = 0
    pbar = tqdm(loader, total=len(loader), position=0, leave=True, desc=f"Validation single steps")
    for data in pbar:

        data = data.to(device)

        # Train with single steps

        loss = singlestep(model, data, train=False, use_noise=False)

        loss = loss.item()
        total_loss += loss
        #if train_step % log_steps == 0:
        #    writer.add_scalar("Training loss per gradient step", loss, train_step)

    return total_loss / len(loader)


box_search_depth = 0.2
box_search_lateral = 0.2
box_search_forward = 0.5

def subsample_func(soilpos, soilvel, wheelpos):

    inds = (soilpos[:,2] > wheelpos[2]-wheel_radius-box_search_depth)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,0] > wheelpos[0]-box_search_forward)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,0] < wheelpos[0]+box_search_forward)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,1] > wheelpos[1]-box_search_lateral)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,1] < wheelpos[1]+box_search_lateral)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    return soilpos, soilvel

