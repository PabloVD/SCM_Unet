from Source.utils import *
#from Source.visualize import *
from tqdm import tqdm

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
        hmap, def_true, glob = data_augmentation(hmap, def_true, glob)
        if use_noise:
            hmap = get_noise_hmap(hmap)

    outsoil = model(hmap, glob)

    if use_log:

        def_true = torch.log(-torch.clamp(def_true,max=0.) + 1.e-8)

    loss_soil = criterion_s(outsoil, def_true)

    return loss_soil

   

# Predict sequence (rollout) starting at timestep in_step with a length rollout_seq and compute loss
def rollout(model, data, in_step=seq_len, rollout_seq=50, optimizer=None, scheduler=None, train=False):

    loss_rol = 0

    maxtimesteps = data.x.shape[2]


    if in_step+rollout_seq>maxtimesteps-1:
        print("max step too large",in_step+rollout_seq-1)

    # SCM
    gnn_out = data.x.clone()    # Afterwards, this is updated!

    # For each time "step" and previous ones, predict step+1
    steps = range(in_step, in_step+rollout_seq)

    batch = data.batch.to(device)
    hmap_init = data.hmap_init.to(device)
    batches = data.wheelpos.shape[0]

    for step in steps:

        glob = data.glob[:,:,step].to(device)
        x_step = gnn_out[:,:,step].to(device)
        wheelpos_step = data.wheelpos[:,:,step].to(device)
        def_true = data.def_true[:,:,step].to(device)
        
        #print(x_step.shape, wheelpos_step[batch].shape, batches)

        relpos = x_step - wheelpos_step[batch]
        windowcond = sampleparts_rectangle(relpos)
        x_step = x_step[windowcond]
        def_true = def_true[windowcond]
        hmap_init_w = hmap_init[windowcond]

        hmap, def_true, pcloud_window, condbox = get_hmap_batched(x_step, wheelpos_step[batch[windowcond]], hmap_init_w, def_true, batches)

        outsoil = model(hmap, glob)

        if use_log:

            def_true = torch.log(-torch.clamp(def_true,max=0.) + 1.e-8)

        loss_soil = criterion_s(outsoil, def_true)
        loss_rol += loss_soil

        #print(outsoil.shape, pcloud_window.shape)

        if use_log:
            outsoil = -torch.exp(outsoil)

        # hmap2pcloud
        pcloud_window[:,2] += outsoil.view(pcloud_window.shape[0])

        x_step[condbox] = pcloud_window

        #print(gnn_out.is_cuda, windowcond.is_cuda, x_step.is_cuda)
        gnn_out[windowcond.to("cpu"),:,step] = x_step.clone().detach().to("cpu")

        if train:

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss_soil
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_soil
                loss.backward()#retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)  # Clear gradients.



    return loss_rol/len(steps), gnn_out



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



# Training routine
def train_rollout(model, loader, optimizer, scheduler, train_step, writer, lastmodelname):
    model.train()

    total_loss = 0
    pbar = tqdm(loader, total=len(loader), position=0, leave=True, desc=f"Training")
    for data in pbar:

        optimizer.zero_grad(set_to_none=True)

        #data = data.to(device)

        # Train with single steps

        
        loss, _ = rollout(model, data, in_step=seq_len, rollout_seq=data.x.shape[2]-seq_len-1, optimizer=optimizer, scheduler=scheduler, train=True)

        
        #if scheduler is not None:
        #    scheduler.step()

        loss = loss.item()
        total_loss += loss
        if train_step % log_steps == 0:
            writer.add_scalar("Training loss per gradient step", loss, train_step)

        #optimizer.zero_grad(set_to_none=True)  # Clear gradients.
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

        loss_rol, gnn_out = rollout(model, data, in_step=seq_len, rollout_seq=maxsteps-seq_len-1)
        #print(gnn_out[:3,:,-3:])
        #print(force_out[:,:,-3:])
        total_loss += loss_rol.item()#/data.x.shape[0]
        #tot_loss_rig += loss_rig.item()#/data.x.shape[0]
        if vis:
            vis_results(data, gnn_out, in_step=0, lensteps=maxsteps-1, force_out=force_out, interval=100)

    #if writer is not None:
    #    compare_truth_pred(data, gnn_out, -1, writer, boundarylist)
        #compare_truth_pred(data, gnn_out, 60, writer, boundarylist)

    return total_loss / len(loader)

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

