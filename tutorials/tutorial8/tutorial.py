import torch
from time import time
#Data generation

def sample_unit_circle(num_points):
    radius=torch.rand(num_points,1)
    angle=torch.rand(num_points,1)*2*torch.pi
    x=radius*torch.cos(angle)
    y=radius*torch.sin(angle)
    data=torch.cat((x,y),dim=1)
    return data



def compute_output(data,theta):
    z=torch.sin(theta*data.reshape(1,-1,2))
    z=z.sum(axis=2)
    return z

def compute_input(data,theta):
    z=torch.exp(torch.cos(theta*data.reshape(1,-1,2)))
    z=z.sum(axis=2)
    return z



theta=torch.rand(1000,1,2)
data_coarse=sample_unit_circle(5000)
theta_coarse=theta.repeat(1,5000,1)
output_coarse=compute_output(data_coarse,theta_coarse).unsqueeze(-1) 
input_coarse=compute_input(data_coarse,theta_coarse).unsqueeze(-1)
##Additional data for superesolution
data_dense=sample_unit_circle(10000)
theta_dense=theta.repeat(1,10000,1)
output_dense=compute_output(data_dense,theta_dense).unsqueeze(-1) 
input_dense=compute_input(data_dense,theta_dense).unsqueeze(-1)



from pina.model import GNO
from pina import Condition,LabelTensor
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.trainer import Trainer

model=GNO(1,1,data_coarse,0.05,inner_size=100)
class GNOSolver(AbstractProblem):
    input_variables=['input']
    input_points=LabelTensor(input_coarse,input_variables)
    output_variables=['output']
    output_points=LabelTensor(output_coarse,output_variables)
    conditions={"data":Condition(input_points=input_points,output_points=output_points)}

problem=GNOSolver()
solver=SupervisedSolver(problem,model,)
trainer=Trainer(solver=solver,max_epochs=5,accelerator='cpu',enable_model_summary=False,batch_size=10)
from pina.loss import LpLoss
loss=LpLoss(2,relative=True)

start_time=time()
trainer.train()
end_time=time()
print(end_time-start_time)