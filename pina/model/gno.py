import torch
import torch.nn as nn
from torch_geometric.nn import pool
from . import FeedForward

class GNO(nn.Module):
    def __init__(self,
                input_features,
                output_features,
                points,
                radius=0.05,
                inner_size=20,
                n_layers=2,
                func=nn.Tanh,
                ):
    
        super().__init__()
        self.input_features=input_features
        self.output_features=output_features
        self.num_points=points.shape[0]
        self.points_size=points.shape[1]
        self.nn=FeedForward(2*self.points_size+2*self.input_features, output_features*output_features, inner_size, n_layers, func)
        self.points=points
        self.neigh=pool.radius(points, points, radius)
        self.linear=nn.Linear(self.input_features,self.output_features,bias=False)
    
    def forward(self, batch):
        points_x=self.points[self.neigh[1]].unsqueeze(0).repeat(batch.shape[0],1,1)
        points_y=self.points[self.neigh[0]].unsqueeze(0).repeat(batch.shape[0],1,1)
        batch_x=batch[:,self.neigh[1],:]
        batch_y=batch[:,self.neigh[0],:]
        new_batch=torch.concatenate((points_x,points_y,batch_x,batch_y),dim=2)
        new_batch=self.nn(new_batch).reshape(batch.shape[0],-1,self.output_features,self.output_features)
        new_batch=torch.matmul(new_batch,batch_y.unsqueeze(-1)).squeeze(-1)
        tmp_list=self.neigh[0].unsqueeze(0).unsqueeze(-1).repeat(batch.shape[0],1,self.output_features)
        tmp_array=torch.zeros(batch.shape[0],batch.shape[1],self.output_features,requires_grad=True)
        lin_part=self.linear(batch)
        k_part=torch.scatter_reduce(tmp_array,1,tmp_list,new_batch,reduce='mean')
        new_batch=lin_part+k_part
        return new_batch
    
    def forward_eval(self,batch,points):
        neigh=pool.radius(points, points, 0.05)
        points_x=points[neigh[1]].unsqueeze(0).repeat(batch.shape[0],1,1)
        points_y=points[neigh[0]].unsqueeze(0).repeat(batch.shape[0],1,1)
        batch_x=batch[:,neigh[1],:]
        batch_y=batch[:,neigh[0],:]
        new_batch=torch.concatenate((points_x,points_y,batch_x,batch_y),dim=2)
        new_batch=self.nn(new_batch).reshape(batch.shape[0],-1,self.output_features,self.output_features)
        new_batch=torch.matmul(new_batch,batch_y.unsqueeze(-1)).squeeze(-1)
        tmp_list=neigh[0].unsqueeze(0).unsqueeze(-1).repeat(batch.shape[0],1,self.output_features)
        tmp_array=torch.zeros(batch.shape[0],batch.shape[1],self.output_features,requires_grad=True)
        lin_part=self.linear(batch)
        k_part=torch.scatter_reduce(tmp_array,1,tmp_list,new_batch,reduce='mean')
        new_batch=lin_part+k_part
        return new_batch




        

