import torch
from . import FeedForward
from torch import nn



class AVNOLayer(nn.Module):
    def __init__(self,hidden_size,func):
        super().__init__()
        self.nn=nn.Linear(hidden_size,hidden_size)
        self.func=func
    
    def forward(self,batch):
        return self.func()(self.nn(batch)+torch.mean(batch,dim=1).unsqueeze(1))

class AVNO(nn.Module):
    def __init__(self,
                input_features,
                output_features,
                points,
                inner_size=100,
                n_layers=4,
                func=nn.GELU,
                ):
    
        super().__init__()
        self.input_features=input_features
        self.output_features=output_features
        self.num_points=points.shape[0]
        self.points_size=points.shape[1]
        self.lifting=FeedForward(input_features+self.points_size,inner_size,inner_size,n_layers,func)
        self.nn=nn.Sequential(*[AVNOLayer(inner_size,func) for _ in range(n_layers)])
        self.projection=FeedForward(inner_size+self.points_size,output_features,output_features,n_layers,func)
        self.points=points
    
    def forward(self, batch):
        points_tmp=self.points.unsqueeze(0).repeat(batch.shape[0],1,1)
        new_batch=torch.concatenate((batch,points_tmp),dim=2)
        new_batch=self.lifting(new_batch)
        new_batch=self.nn(new_batch)
        new_batch=torch.concatenate((new_batch,points_tmp),dim=2)
        new_batch=self.projection(new_batch)
        return new_batch
    
    def forward_eval(self,batch,points):
        points_tmp=points.unsqueeze(0).repeat(batch.shape[0],1,1)
        new_batch=torch.concatenate((batch,points_tmp),dim=2)
        new_batch=self.lifting(new_batch)
        new_batch=self.nn(new_batch)
        new_batch=torch.concatenate((new_batch,points_tmp),dim=2)
        new_batch=self.projection(new_batch)
        return new_batch




        

