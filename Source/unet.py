import torch
from torch import nn
torch.backends.cudnn.benchmark = True

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, global_emb_dim, up=False):
        super().__init__()
        self.global_mlp =  nn.Linear(global_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, glob):

        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        global_emb = self.relu(self.global_mlp(glob))
        # Extend last 2 dimensions
        #global_emb = global_emb[(..., ) + (None, ) * 2]
        global_emb = global_emb.unsqueeze(-1).unsqueeze(-1)
        # Add time channel
        h = h + global_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        h = self.transform(h)
        return h


# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         # TODO: Double check the ordering here
#         return embeddings


class Unet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, input_channels = 1, num_layers = 5, hidden_channels_in = 32, global_emb_dim = 8):
        super().__init__()
        
        self.input_channels = input_channels

        #down_channels = (64, 128, 256, 512, 1024)
        #up_channels = (1024, 512, 256, 128, 64)
        down_channels = [2**i*hidden_channels_in for i in range(num_layers)]
        up_channels = list(reversed(down_channels))
        out_dim = 1 

        # Time embedding
        self.global_mlp = nn.Sequential(
                nn.Linear(global_emb_dim, global_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(input_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    global_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        global_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, glob):
        # Embedd time
        t = self.global_mlp(glob)
        # Initial conv
        #print("First",x.shape)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            #print(x.shape)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            #print(residual_x.shape)
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
            #print(x.shape)
        return self.output(x)
    



if __name__=="__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    device = "cuda"

    model = Unet(input_channels = 3, num_layers = 3, hidden_channels_in = 32, global_emb_dim = 6)
    model = model.to(device)
    #model = model.half()

    time_tot = []

    for t in range(10000):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = torch.randn((4,3,12,12))
        glob = torch.randn((4,6))
        x = x.to(device)
        glob = glob.to(device)    
        #x, glob = x.half(), glob.half()

        out = model(x, glob)
        out = out.to("cpu")

        end.record()
        torch.cuda.synchronize()
        time_infer = start.elapsed_time(end)
        time_tot.append( time_infer )

    burnphase = 100
    time_tot = np.array(time_tot)
    time_tot = time_tot[burnphase:]
    bins = 100

    plt.figure()
    plt.hist(time_tot, bins=bins )
    plt.title("Mean time: {:.1e} +- {:.1e} ms".format(time_tot.mean(), time_tot.std()))
    #plt.yscale("log")
    plt.xlabel("Time [ms]")
    plt.savefig("time.png")

