import torch
from time import time
from pina.model import GNO
from pina import Condition,LabelTensor
from pina.problem import AbstractProblem
from pina.solvers import SupervisedSolver
from pina.trainer import Trainer
from pina.loss import LpLoss

#Data generation

torch.manual_seed(0)

def sample_unit_circle(num_points):
    radius=torch.rand(num_points,1)
    angle=torch.rand(num_points,1)*2*torch.pi
    x=radius*torch.cos(angle)
    y=radius*torch.sin(angle)
    data=torch.cat((x,y),dim=1)
    return data

#sin(a*x+b*y)
def compute_input(data,theta):
    data=data.reshape(1,-1,2)
    z=torch.sin(theta[:,:,0]*data[:,:,0]+theta[:,:,1]*data[:,:,1])
    return z

#1+convolution of sin(a*x+b*y) with sin(x) over [0,2pi]x[0,2pi    ]
def compute_output(data,theta):
    data=data.reshape(1,-1,2)
    z=1-4*torch.sin(torch.pi*theta[:,:,0])*torch.sin(torch.pi*theta[:,:,1])*torch.cos(theta[:,:,0]*(torch.pi*data[:,:,0])+theta[:,:,1]*(torch.pi*data[:,:,1]))/((theta[:,:,0]**2-1)*theta[:,:,1])
    return z



theta=1+0.01*torch.rand(300,1,2)
data_coarse=sample_unit_circle(1000)
output_coarse=compute_output(data_coarse,theta).unsqueeze(-1) 
input_coarse=compute_input(data_coarse,theta).unsqueeze(-1)
data_dense=sample_unit_circle(1000)
output_dense=compute_output(data_dense,theta).unsqueeze(-1) 
input_dense=compute_input(data_dense,theta).unsqueeze(-1)


model=GNO(1,1,data_coarse,inner_size=500,n_layers=4)
class GNOSolver(AbstractProblem):
    input_variables=['input']
    input_points=LabelTensor(input_coarse,input_variables)
    output_variables=['output']
    output_points=LabelTensor(output_coarse,output_variables)
    conditions={"data":Condition(input_points=input_points,output_points=output_points)}

batch_size=1
problem=GNOSolver()
solver=SupervisedSolver(problem,model,optimizer_kwargs={'lr':1e-3},optimizer=torch.optim.AdamW)
trainer=Trainer(solver=solver,max_epochs=5,accelerator='cpu',enable_model_summary=False,batch_size=batch_size)
loss=LpLoss(2,relative=True)

start_time=time()
trainer.train()
end_time=time()
print(end_time-start_time) 
solver.neural_net=solver.neural_net.eval()
loss=torch.nn.MSELoss()
num_batches=len(input_coarse)//batch_size
num=0
dem=0
for i in range(num_batches):
    input_variables=['input']
    myinput=LabelTensor(input_coarse[i].unsqueeze(0),input_variables)
    tmp=model(myinput).detach().squeeze(0)
    num=num+torch.linalg.norm(tmp-output_coarse[i])**2
    dem=dem+torch.linalg.norm(output_coarse[i])**2
print("Training mse loss is", torch.sqrt(num/dem))


num=0
dem=0
for i in range(num_batches):
    input_variables=['input']
    myinput=LabelTensor(input_dense[i].unsqueeze(0),input_variables)
    tmp=model.forward_eval(myinput,data_dense).detach().squeeze(0)
    num=num+torch.linalg.norm(tmp-output_dense[i])**2
    dem=dem+torch.linalg.norm(output_dense[i])**2
print("Super Resolution mse loss is", torch.sqrt(num/dem))
