import torch
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList, Embedding, SiLU, ModuleDict, BatchNorm1d
#from torch_geometric.nn import MetaLayer, LayerNorm, GraphNorm
from torch_geometric.nn import LayerNorm, BatchNorm, InstanceNorm, GraphNorm
#from torch_scatter import scatter_add
from torch_scatter import scatter
#import torch.utils.checkpoint
#from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch import Tensor
from typing import Optional, Tuple
from Source.cnn import HmapNet

#use_checkpoints = False

# Get memory of a tensor in Mb
def sizetensor(tensor):
    return tensor.element_size()*tensor.nelement()/1.e6

# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/meta.html#MetaLayer
class MyMetaLayer(torch.nn.Module):

    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(
            self, x: Tensor, edge_index: Tensor,
            edge_attr: Tensor, u: Tensor,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')


def standarize(vec, vecmean, vecstd):
    return (vec - vecmean)/vecstd

def dotprod2d(vec1,vec2):
    #print(vec1.shape, vec2.shape)
    return torch.bmm(vec1.view(-1,1,2),vec2.view(-1,2,1)).squeeze()

# decompose vector in components parallel, orthogonal to the wheel orientation, and vertical
def vec_orientframe(diff, orient2d, normal2d):
    diff2d = diff[:,:2]
    diff_z = diff[:,2:3]
    diff_par = dotprod2d(diff2d,orient2d)
    diff_ort = dotprod2d(diff2d,normal2d)
    vecframe = torch.cat([diff_par.view(-1,1), diff_ort.view(-1,1), diff_z], dim=1)
    return vecframe

def get_ort2d(orient2d, rot90):
    #norm2d = torch.matmul(orient2d, rot90.T.view(2,2,1)).T.view(-1,2)
    #"""
    norm2d = torch.matmul(rot90,orient2d[0]).view(1,-1)
    for i in range(1,orient2d.shape[0]):
        norm2d = torch.cat([norm2d, torch.matmul(rot90,orient2d[i]).view(1,-1)])
    #"""
    #return norm2d
    return norm2d

def normlayer(norm, node_out):

    if norm=="instance":
        return InstanceNorm(node_out)
    elif norm=="layer":
        return LayerNorm(node_out)
    elif norm=="batch":
        return BatchNorm(node_out)
    elif norm=="graph":
        return GraphNorm(node_out)
    else:
        print("No valid normalization layer provided")

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=None):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  ReLU(),
                  #Linear(hid_channels, hid_channels),   # largermlps
                  #ReLU(),
                  #Linear(hid_channels, hid_channels),   # largermlps
                  #ReLU(),
                  Linear(hid_channels, edge_out)]

        #if finalrelu:
        #    layers += []
        #layers.append(ReLU())
        #if self.norm:  layers.append(LayerNorm(node_out))
        #if self.norm:  layers.append(BatchNorm(node_out))
        if self.norm is not None:  layers.append(normlayer(norm, node_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src: Tensor, dest: Tensor, edge_attr: Tensor, u: Tensor, batch: Optional[Tensor]):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        #out = torch.utils.checkpoint.checkpoint_sequential(self.edge_mlp, 2, out)
        if self.residuals:
            out = out + edge_attr
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, glob_feats, residuals=True, norm=None):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + edge_out + glob_feats, hid_channels),
                  ReLU(),
                  #Linear(hid_channels, hid_channels),   # largermlps
                  #ReLU(),
                  #Linear(hid_channels, hid_channels),   # largermlps
                  #ReLU(),
                  Linear(hid_channels, node_out)]
                  
        #layers.append(ReLU())
        #if self.norm:  layers.append(LayerNorm(node_out))
        #if self.norm:  layers.append(BatchNorm(node_out))
        if self.norm is not None:  layers.append(normlayer(norm, node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor, batch: Optional[Tensor]):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index[0], edge_index[1]
        #out = torch.cat([x[row], edge_attr], dim=1)
        #out = self.node_mlp_1(out)
        #out = edge_attr
        #out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = scatter(edge_attr, col, dim=0, reduce="sum", dim_size=x.size(0))
        #out = scatter(edge_attr, col, dim=0, reduce="mean", dim_size=x.size(0))
        #out = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        #out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        #out = torch.cat([x, out, u[batch]], dim=1)
        #out = torch.cat([x, out], dim=1)

        out = torch.cat([x, out, u[batch]], dim=1)
        #out = self.node_mlp(out)
        #if use_checkpoints:
        #    out = torch.utils.checkpoint.checkpoint_sequential(self.node_mlp, 2, out)
        #else:
        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out

# Graph Neural Network architecture, based on the Interaction Network (arXiv:1612.00222, arXiv:2002.09405)
class GNN(torch.nn.Module):
    def __init__(self, n_layers, hidden_channels, linkradius, device, num_particle_types=2, embeddings=4, use_torque=False, use_open3d=False, equi=True, use_throttle = False, use_wheeltype = False, use_boundary = False, debug = False, use_sinkage = True, use_vel = True, use_norm=None, use_hmap=False, use_wheelnodes=True, use_relposwheel=False):
        super().__init__()

        self.use_skip = False
        self.n_layers = n_layers
        self.loop = False
        self.dim = 3
        self.dim_out = 1
        self.linkradius = linkradius
        self.device = device
        self.num_particle_types = num_particle_types
        if embeddings is None:
            self.part_type_embedding_size = self.num_particle_types
        else:
            self.part_type_embedding_size = embeddings
        self.wheel_embed = 8    # num embedings for wheel type
        self.rot90 = torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float32,device=self.device)
        self.use_sinkage = use_sinkage
        self.use_vel = use_vel
        self.use_hmap = use_hmap
        hmap_layers = 64
        self.hmap_out = torch.zeros(0,dtype=torch.float32,device=self.device)   # default, not used

        # If true, predict torque (3d vector) for rigid body, otherwise predict rotation quaternion (4 components)
        self.use_torque = use_torque
        # Use equivariant formulation if true
        self.equi = equi
        self.use_throttle = use_throttle
        self.num_soils = 1  # Number of different types of soils
        self.use_wheeltype = use_wheeltype
        self.use_boundary = use_boundary
        self.use_norm = use_norm
        self.debug = debug
        self.use_wheelnodes = use_wheelnodes
        self.use_relposwheel = use_relposwheel

        # Normalizations
        #"""
        # Global frame
        if self.equi:
            self.linmean = torch.tensor([1.8929, -0.0187, -0.1054],dtype=torch.float32,device=self.device).view(1,3)
            self.linstd = torch.tensor([1.7907, 0.2195, 0.5747],dtype=torch.float32,device=self.device).view(1,3)
            self.angmean = torch.tensor([1.2358e-03, 1.1965e+01, 4.9697e-02],dtype=torch.float32,device=self.device).view(1,3)
            self.angstd = torch.tensor([0.5090, 18.1045,  2.2802],dtype=torch.float32,device=self.device).view(1,3)
        else:
            self.linmean = torch.tensor([1.8959,  0.1170, -0.0963],dtype=torch.float32,device=self.device).view(1,3)
            self.linstd = torch.tensor([1.7463, 0.5545, 0.5613],dtype=torch.float32,device=self.device).view(1,3)
            self.angmean = torch.tensor([-0.7546, 11.9311,  0.0504],dtype=torch.float32,device=self.device).view(1,3)
            self.angstd = torch.tensor([4.3460, 17.8348,  2.3010],dtype=torch.float32,device=self.device).view(1,3)
        self.modlinmean = torch.tensor(2.0869,dtype=torch.float32,device=self.device).view(1,1)
        self.modlinstd = torch.tensor(1.7130,dtype=torch.float32,device=self.device).view(1,1)
        self.modangmean = torch.tensor(12.5134,dtype=torch.float32,device=self.device).view(1,1)
        self.modangstd = torch.tensor(18.1274,dtype=torch.float32,device=self.device).view(1,1)
        #"""
        """
        self.linnorm = torch.tensor([1.3, 1.3, 0.3],dtype=torch.float32,device=self.device).view(1,3)
        self.angnorm = torch.tensor([10., 10., 3.],dtype=torch.float32,device=self.device).view(1,3)
        self.modlinnorm = torch.tensor(1.8,dtype=torch.float32,device=self.device).view(1,1)
        self.modangnorm = torch.tensor(10.,dtype=torch.float32,device=self.device).view(1,1)
        """

        self.partembed_mean = torch.tensor([1.10802382e-01, 2.19168607e-04, -7.06245599e-04, -1.46817835e-02],dtype=torch.float32,device=self.device).view(1,self.part_type_embedding_size)
        self.partembed_std = torch.tensor([1.3619718e-01, 3.1031859e-03, 1.7684922e-03, 6.6883272e-01],dtype=torch.float32,device=self.device).view(1,self.part_type_embedding_size)
        self.velnodes_mean = torch.tensor([1.12740628e-01, -2.79903300e-02, 2.27844405e+00],dtype=torch.float32,device=self.device).view(1,3)
        self.velnodes_std = torch.tensor([5.9505053e+00, 6.4784640e-01, 4.5896940e+00],dtype=torch.float32,device=self.device).view(1,3)
        self.edgefeats_mean = torch.tensor([-3.5669328e-12, -2.1009533e-11, -4.5002162e-12, 8.6174250e-01],dtype=torch.float32,device=self.device).view(1,4)     # change if radius larger
        self.edgefeats_std = torch.tensor([0.56437975, 0.54960686, 0.36757547, 0.11447129],dtype=torch.float32,device=self.device).view(1,4)
        self.wheelembed_mean = torch.tensor([-1.12705715e-01, 4.51746128e-05, 7.20329762e-01, 1.34982124e-01, 9.44461003e-02, -9.08359051e-01, 1.34332944e-03, -3.73212904e-01],dtype=torch.float32,device=self.device).view(1,self.wheel_embed)
        self.wheelembed_std = torch.tensor([0.22337762, 0.00543787, 0.83979475, 0.44682184, 0.07505064, 0.8946016, 0.00493907, 0.44667953],dtype=torch.float32,device=self.device).view(1,self.wheel_embed)
        


        # Input node features: seq_len velocities + boundaries
        #node_in = seq_len*dim + 2*dim + self.part_type_embedding_size
        #node_in = seq_len*dim + self.part_type_embedding_size
        # SCM
        node_in = 0
        if self.use_wheelnodes:
            node_in += self.part_type_embedding_size
        if self.use_boundary:
            node_in += 1
        if self.use_sinkage:
            node_in += 1
        if use_vel:
            node_in += 3
        if self.use_relposwheel:
            node_in += 3

        # sinkageprev
        #node_in += 1

        # Input edge features: [p_i-p_j, |p_i-p_j|]
        edge_in = self.dim + 1
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        glob_feats = 8  # 3 wheel linear velocity plus modulus + 3 wheel angular velocity plus modulus
        # nolinangglob
        #glob_feats = 0
        if self.use_throttle:
            glob_feats += 3
        if self.use_wheeltype:
            glob_feats += self.wheel_embed
        if self.use_hmap:
            #glob_feats += hmap_layers
            glob_feats += hmap_layers*2

        if self.use_torque:
            self.rigdof = 6        # 3 force of c.o.m., plus 3 torque
        else:
            self.rigdof = 3        # 3 force of c.o.m.

        # Define architecture

        layers = []

        # Encoder layer
        inlayer = MyMetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, glob_feats, residuals=False, norm=self.use_norm),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False))
        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Decoder layers
        n_decoder_layers = 4

        # Hidden graph layers
        for i in range(n_layers):

            lay = MyMetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, glob_feats, norm=self.use_norm),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels))
            layers.append(lay)

        # Decoder
        """node_out = dim_out

        outlayer = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False),
                             edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False))
        layers.append(outlayer)"""

        self.layers = ModuleList(layers)

        self.soil_decoder = []
        self.wheel_decoder = []

        for soil in range(self.num_soils):

            # New decoder
            #"""
            soillayer = Sequential(Linear(node_out, hid_channels),
                                  ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels, hid_channels),
                                  ReLU(),
                                  #Linear(hid_channels, hid_channels),
                                  #ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels, hid_channels//2),
                                  ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels//2, self.dim_out))

            self.soil_decoder.append(soillayer)
            #"""
            # Modification13Jan
            #riglayer = Sequential(Linear(node_out + glob_feats, hid_channels),
            #riglayer = Sequential(Linear(node_out + glob_feats + edge_out, hid_channels),
            #riglayer = Sequential(Linear(edge_out, hid_channels),
            # Modification16Jan
            """
            riglayer = Sequential(Linear(node_out, hid_channels),
            #riglayer = Sequential(Linear(node_out + glob_feats, hid_channels),
                                  ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels, hid_channels),
                                  ReLU(),
                                  #Linear(hid_channels, hid_channels),
                                  #ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels, hid_channels//2),
                                  ReLU(),
                                  #SiLU(),
                                  Linear(hid_channels//2, self.rigdof))
            #"""
            # Mod dec
            #"""
            rig_layers = []
            for i in range(n_decoder_layers - 1):
                rig_layers += [Linear(node_out, hid_channels), ReLU()]
                node_out = hid_channels
                if i>=(n_decoder_layers - 1 -3):
                    hid_channels //= 2
            rig_layers += [Linear(hid_channels*2, self.rigdof)]
            riglayer = Sequential(*rig_layers)
            #"""


            self.wheel_decoder.append(riglayer)

        self.soil_decoder = ModuleList(self.soil_decoder)
        self.wheel_decoder = ModuleList(self.wheel_decoder)
        #self.soil_decoder = ModuleDict(self.soil_decoder)
        #self.wheel_decoder = ModuleDict(self.wheel_decoder)



        self.part_type_embedding = Embedding(self.num_particle_types, self.part_type_embedding_size)

        #if self.use_wheeltype:
        self.wheel_embedding = Embedding(4, self.wheel_embed)

        """
        self.hiddenacts = {
            "Node in" : [], 
            "Edge in" : [], 
            "Global in" : [], 
            "Node hidden out" : [], 
            }
        """
        self.hiddenacts = {
            "Node in" : torch.empty(0,7), 
            "Edge in" : torch.empty(0,4), 
            "Global in" : torch.empty(0,glob_feats) 
            #Node hidden out" : torch.empty(0,hidden_channels,device=self.device)
            }
        
        for ii in range(len(layers)):
            self.hiddenacts["Node out "+str(ii)] = torch.empty(0,hidden_channels)


        """
        if self.use_hmap:
            self.hmap_model = Sequential(Linear(1, hmap_layers//8),
                                    ReLU(),
                                    Linear(hmap_layers//8, hmap_layers//4),
                                    ReLU(),
                                    Linear(hmap_layers//4, hmap_layers//2),
                                    ReLU(),
                                    Linear(hmap_layers//2, hmap_layers))
        """
        if self.use_hmap:
            self.hmap_model = HmapNet(mid_channels=hmap_layers)




        # benchmark
        """
        #inchanforce = 6 # sinkageprev
        inchanforce = 5
        self.forcenet = Sequential(Linear(inchanforce, hid_channels//2),
                                ReLU(),
                                Linear(hid_channels//2, hid_channels),
                                ReLU(),
                                Linear(hid_channels, hid_channels),
                                ReLU(),
                                Linear(hid_channels, hid_channels),
                                ReLU(),
                                Linear(hid_channels, hid_channels),
                                ReLU(),
                                Linear(hid_channels, hid_channels//2),
                                ReLU(),
                                Linear(hid_channels//2, hid_channels//4),
                                ReLU(),
                                Linear(hid_channels//4, self.rigdof))
        #"""

        """
        # Radius search to build the graphs
        self.use_open3d = use_open3d
        if self.use_open3d:
            import open3d.ml.torch as ml3d
            self.nsearch = ml3d.layers.FixedRadiusSearch(return_distances=False, ignore_query_point=True)

        """
    """


    # Get distance from boundaries
    def boundary_distance(self, position):

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        # position with shape (n_nodes, 2)
        distance_to_lower_boundary = (position - self.boundaries[:, 0])
        distance_to_upper_boundary = (self.boundaries[:, 1] - position)
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self.linkradius, -1., 1.)
        return normalized_clipped_distance_to_boundaries

    def get_edges(self, points, radius, batch=None):

        if batch is None:
            batch = points.new_zeros(points.size(0), dtype=torch.long)

        bounds_max = torch.max(points,dim=0).values.view(1,-1)
        bounds_min = torch.min(points,dim=0).values.view(1,-1)

        points = points + (bounds_max - bounds_min + radius*2.)*batch.view(-1,1)

        neighsearch = self.nsearch(points, points, radius)


        splits = neighsearch.neighbors_row_splits
        num_neighs = splits[1:] - splits[:-1]

        numneighs = num_neighs[num_neighs>0]

        querypoints = num_neighs.nonzero(as_tuple=False).view(-1)
        row = neighsearch.neighbors_index.view(1,-1)
        col = torch.repeat_interleave(querypoints, numneighs).view(1,-1)

        edge_index = torch.cat([row,col],dim=0)

        return edge_index

    """
    def embed_hmap(self, hmap):
        #hmap = data.hmap
        #numsoilparts = data.numsoilparts
        """
        batch_hmap = torch.empty(0,device=self.device,dtype=int)
        for b in range(numsoilparts.shape[0]):
            batch_hmap = torch.cat([batch_hmap, b*torch.ones(numsoilparts[b,0],device=self.device,dtype=int)])
        print(hmap.shape)
        hmap_out = self.hmap_model(hmap/self.linkradius)
        hmap_out = scatter(hmap_out, batch_hmap, dim=0, reduce="max")
        self.hmap_out = hmap_out
        """
        
        hmap_out = self.hmap_model(hmap)
        self.hmap_out = hmap_out.view(hmap_out.shape[0],-1)

    # Get graph in global frame
    def get_graph(self, x, batch, part_types):

        pos = x[:,:self.dim]
        sink = x[:,self.dim:self.dim+1]
        vel = x[:,1+self.dim:1+self.dim+3]
        
        h = torch.empty((x.shape[0],0),dtype=torch.float32,device=self.device)

        if self.use_wheelnodes:
            partembed = self.part_type_embedding(part_types)
            partembed = standarize(partembed, self.partembed_mean, self.partembed_std)
            h = torch.cat([h, partembed], dim=1)

        if self.use_vel:
            vel = standarize(vel, self.velnodes_mean, self.velnodes_std)
            h = torch.cat([h, vel], dim=1)

        if self.use_sinkage:
            h = torch.cat([h, sink/self.linkradius], dim=1)

        if self.use_relposwheel:
            relpos = x[:,7:10]
            h = torch.cat([h, relpos/self.linkradius], dim=1)
            
        # Get edges

        #pos, edge_index = data.pos, data.edge_index
        """
        if self.use_open3d:
            edge_index = self.get_edges(pos, self.linkradius, batch=batch)
        else:
            edge_index = radius_graph(pos, r=self.linkradius, batch=batch, loop=self.loop)
        """
        edge_index = radius_graph(pos, r=self.linkradius, batch=batch, loop=self.loop)

        row, col = edge_index[0], edge_index[1]

        # Edge features
        diff = (pos[row]-pos[col])/self.linkradius
        dist = torch.norm(diff, dim=1, keepdim=True)
        edge_attr = torch.cat([diff, dist], dim=1)

        return h, edge_index, edge_attr

    #"""

    # # Get distance from the bottom boundary
    # def boundary_distance(self, position, boundaryz, batch):

    #     distbound = position[:,2] - boundaryz[batch].view(-1)
    #     normalized_clipped_distance_to_boundary = torch.clamp(distbound / self.linkradius, -1., 1.)

    #     return normalized_clipped_distance_to_boundary.view(position.shape[0],1)

    # # Soft equivariance: vector magnitudes relative to orientation of the wheel
    # def get_graph_equi(self, x, batch, part_types, orient2d, normal2d):

    #     pos = x[:,:self.dim]
    #     vel = x[:,1+self.dim:1+self.dim+3]

    #     orientbatch = orient2d[batch]
    #     normalbatch = normal2d[batch]

    #     # Node features
    #     #h = x[:,self.dim:]
    #     #h = self.boundary_distance(pos)
    #     #print(torch.min(h))

    #     partembed = self.part_type_embedding(part_types)
    #     partembed = standarize(partembed, self.partembed_mean, self.partembed_std)
    #     #h = torch.cat([h, partembed], dim=1)
    #     h = partembed

    #     # Velocity features
    #     """
    #     for seqstep in range(self.seq_len):
    #         # Velocity vector for timestep seqstep
    #         vel = torch.cat([vels[:,0+seqstep].reshape(-1,1), vels[:,self.seq_len+seqstep].reshape(-1,1), vels[:,2*self.seq_len+seqstep].reshape(-1,1) ], dim=1)
    #         h = torch.cat([h, vec_orientframe(vel, orientbatch, normalbatch)], dim=1)
    #     """
    #     # SCM
    #     if self.use_vel:
    #         velframe = vec_orientframe(vel, orientbatch, normalbatch)
    #         velframe = standarize(velframe, self.velnodes_mean, self.velnodes_std)
    #         h = torch.cat([h, velframe], dim=1)

    #     # Get edges

    #     #pos, edge_index = data.pos, data.edge_index
    #     #if self.use_open3d:
    #     #    edge_index = self.get_edges(pos, self.linkradius, batch=batch)
    #     #else:
    #     #    edge_index = radius_graph(pos, r=self.linkradius, batch=batch, loop=self.loop)
    #     edge_index = radius_graph(pos, r=self.linkradius, batch=batch, loop=self.loop)
    #     #print(edge_index.type())
    #     #print(edge_index.shape[1]/pos.shape[0])

    #     row, col = edge_index[0], edge_index[1]
    #     # positions in wheel frame
    #     pos = vec_orientframe(pos, orientbatch, normalbatch)

    #     # Edge features
    #     diff = (pos[row]-pos[col])/self.linkradius
    #     dist = torch.norm(diff, dim=1, keepdim=True)
    #     #print(vec_orientframe(diff, orient2d, normal2d).shape)
    #     edge_attr = torch.cat([diff, dist], dim=1)

    #     return h, edge_index, edge_attr


    #def forward(self, data):
    def forward(self, dataseq, batch, part_types, glob, wheeltype, soiltype, wheelframe):

        #orient2d = wheelframe
        #orient2d = glob[:,:2]/torch.norm(glob[:,:2],dim=1,keepdim=True)
        #normal2d = get_ort2d(orient2d, self.rot90)

        """
        if self.equi:
            h, edge_index, edge_attr = self.get_graph_equi(dataseq, batch, part_types, orient2d, normal2d)
        else:
            h, edge_index, edge_attr = self.get_graph(dataseq, batch, part_types)
        #"""
        h, edge_index, edge_attr = self.get_graph(dataseq, batch, part_types)

        edge_attr = standarize(edge_attr, self.edgefeats_mean, self.edgefeats_std)

        


        condrig = (part_types==1)
        condsoil = ~condrig
        batchsoil = batch[condsoil]

        
        # Global feature vehicle velocity (to be changed in chrono)
        
        wheeltypeembed = self.wheel_embedding(wheeltype.view(-1))



        linvel = glob[:,:3]
        modlinvel = torch.norm(linvel, dim=1, keepdim=True)
        modlinvel = standarize(modlinvel, self.modlinmean, self.modlinstd)

        # if self.equi:
        #     linvel = vec_orientframe(linvel, orient2d, normal2d)

        linvel = standarize(linvel, self.linmean, self.linstd)

        angvel = glob[:,3:6]
        modangvel = torch.norm(angvel, dim=1, keepdim=True)
        modangvel = standarize(modangvel, self.modangmean, self.modangstd)

        # if self.equi:
        #     angvel = vec_orientframe(angvel, orient2d, normal2d)

        angvel = standarize(angvel, self.angmean, self.angstd)

        
        u = torch.cat([linvel, modlinvel, angvel, modangvel], dim=1)

        if self.use_wheeltype:
            wheeltypeembed = standarize(wheeltypeembed, self.wheelembed_mean, self.wheelembed_std)
            u = torch.cat([u, wheeltypeembed], dim=1)

        #print(u)
        if self.use_throttle:
            u = torch.cat([u, glob[:,8:11]], dim=1)

        if self.use_hmap:
            #print(u.shape, self.hmap_out.shape)
            u = torch.cat([u, self.hmap_out], dim=1)

        if self.debug:
            self.hiddenacts["Node in"] = torch.cat([self.hiddenacts["Node in"], h.cpu().detach()],dim=0)
            self.hiddenacts["Edge in"] = torch.cat([self.hiddenacts["Edge in"], edge_attr.cpu().detach()],dim=0)
            self.hiddenacts["Global in"] = torch.cat([self.hiddenacts["Global in"], u.cpu().detach()],dim=0)

        #for layer in self.layers:
        for i, layer in enumerate(self.layers):

            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, batch)

            if self.debug:
                self.hiddenacts["Node out "+str(i)] = torch.cat([self.hiddenacts["Node out "+str(i)], h.cpu().detach()],dim=0)

       

        # Soil
        soil = h[condsoil]

        stypebatchsoil = soiltype[batchsoil].view(-1)

        outsoil = torch.zeros((soil.shape[0],3),device=self.device)

        for soilind, layerdecoder in enumerate(self.soil_decoder):

            condsoil = (stypebatchsoil == soilind)

            outsoil = layerdecoder(soil[condsoil])

        


        return outsoil



def data_prev_steps_provisional(pos_seq, seq_len):

    #if step+1>maxtimesteps:
    #    print("warning, end of array reached")
    #if step-seq_len<0:
    #    print("warning, begining of array reached")

    #pos_seq = x[:,:,step-seq_len:step+1]  # index 0 is initial step, -1 is last instant, current step
    #pos_seq = x[:,:,step:step+seq_len+1]

    vel_seq = pos_seq[:,:,1:] - pos_seq[:,:,:-1]  # index 0 is vel in initial step, -1 is vel in last step

    #vel_seq = (vel_seq-vel_mean.view(-1,1))/vel_std.view(-1,1)#.to(device)
    vel_seq = vel_seq#/vel_std

    """last_pos = pos_seq[:,:,-1].view(-1,dim,1)

    new_seq = torch.cat([last_pos,vel_seq],dim=2)

    new_seq = new_seq.view(pos_seq.shape[0],-1)"""

    vel_seq = vel_seq.reshape(pos_seq.shape[0],3*seq_len)

    last_pos = pos_seq[:,:,-1]
    new_seq = torch.cat([last_pos,vel_seq],dim=1)

    return new_seq


if __name__=="__main__":

    #vec1,vec2 = torch.rand(25, 2),torch.rand(25, 2)

    #print(dotprod2d(vec1,vec2).shape)
    boundarylist = [[-100., 100.], [-100., 100.], [-100., 100. ]]
    boundaries = torch.tensor(boundarylist, requires_grad=False).float()

    # Linking radius
    linkradius = 1.
    # Number of hidden features in each MLP
    hidden_channels = 128
    # Number of intermediate message passing layers
    n_layers = 8
    seq_len = 2


    model = GNN(n_layers=n_layers,
                hidden_channels=hidden_channels,
                linkradius=linkradius,
                dim=3,
                boundaries=boundaries,
                device="cpu",
                seq_len=seq_len,
                dim_out=3,
                num_particle_types=2,
                embeddings=4)


    batch = torch.cat([torch.zeros(10,dtype=torch.int64), torch.ones(5,dtype=torch.int64)])
    part_types = torch.cat([torch.zeros(7,dtype=torch.int64), torch.ones(3,dtype=torch.int64), torch.zeros(2,dtype=torch.int64), torch.ones(3,dtype=torch.int64)])
    step = 5

    pos = torch.randn((15,3,6))
    #posprev = torch.randn((15,3))
    #posprevprev = torch.randn((15,3))
    pos_seq = pos[:,:,step-seq_len:step+1]
    #print(pos_seq.shape)
    dataseq = data_prev_steps_provisional(pos_seq, seq_len)

    #dataseq = torch.randn((15,3+3*seq_len))
    glob = torch.tensor([[1,0],[1,0]],dtype=torch.float32)

    """
    normm = torch.tensor([[0,-1],[0,-1]],dtype=torch.float32)
    print(torch.norm(pos[:,:,0],dim=1))
    newpos=vec_orientframe(pos[:,:,0], glob[batch], normm[batch])
    print(torch.norm(newpos,dim=1))
    exit()
    """

    outsoil, outrig = model(dataseq, batch, part_types, glob)
    #print(outsoil.shape, outrig.shape)

    print("Rotate outputs")

    # Rotate inputs
    ang = torch.tensor(1.5)
    RotM = torch.tensor([[torch.cos(ang), torch.sin(ang),0],[-torch.sin(ang),torch.cos(ang),0],[0,0,1]])

    rotatedsoil = torch.matmul(RotM,outsoil.T).T
    print(rotatedsoil[:4])
    rotatedacc = torch.matmul(RotM,outrig[:,:3].T).T
    print(rotatedacc[:4])
    rotatedtorq = outrig[:,3:7]
    rotatedtorq[:,1:] = torch.matmul(RotM,outrig[:,4:7].T).T
    print(rotatedtorq[:4])

    print("Rotate inputs, then get outputs")


    #for seq in range(seq_len+1):
    #    print(dataseq[0,seq:seq+3])
    #    dataseq[:,seq:seq+3] = torch.matmul(RotM,dataseq[:,seq:seq+3].T).T

    for t in range(pos.shape[2]):
        pos[:,:,t] = torch.matmul(RotM,pos[:,:,t].T).T

    pos_seq = pos_seq = pos[:,:,step-seq_len:step+1]
    dataseq = data_prev_steps_provisional(pos_seq, seq_len)

    for g in range(len(glob)):
        glob[g] = torch.matmul(RotM[:2,:2],glob[g])

    outsoil, outrig = model(dataseq, batch, part_types, glob)
    print(outsoil[:4])
    print(outrig[:,:3])
    print(outrig[:,3:])

    exit()
    orient2d = torch.tensor([[1,0],[1,0],[1,0],[1,0]],dtype=torch.float32)
    rot90 = torch.tensor([[0,1],[-1,0]],dtype=torch.float32)
    #rot90 = rot90.repeat_interleave(4,dim=0,output_size=(4,2,2))
    print(orient2d.shape, rot90.shape)
    print(rot90)
    norm2d = torch.matmul(rot90,orient2d[0]).view(1,-1)
    for i in range(1,orient2d.shape[0]):
        norm2d = torch.cat([norm2d, torch.matmul(rot90,orient2d[i]).view(1,-1)])
    print(orient2d, norm2d)
