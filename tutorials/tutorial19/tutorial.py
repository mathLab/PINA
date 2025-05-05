#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Data structure for SciML: `Tensor`, `LabelTensor`, `Data` and `Graph`
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial19/tutorial.ipynb)
#
# In this tutorial, we’ll quickly go through the basics of Data Structures for Scientific Machine Learning, convering:
# 1. **PyTorch Tensors** / **PINA LabelTensors**
# 2. **PyTorch Geometric Data** / **PINA Graph**
#
# first let's import the data structures we will use!

# In[ ]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab[tutorial]"')

import warnings
import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

from pina import LabelTensor, Graph


# ## PyTorch Tensors
#
# A **tensor** is a multi-dimensional matrix used for storing and manipulating data in PyTorch. It's the basic building block for all computations in PyTorch, including deep learning models.
#
# You can create a tensor in several ways:

# In[2]:


# Creating a tensor from a list
tensor_1 = torch.tensor([1, 2, 3, 4])
print(tensor_1)

# Creating a tensor of zeros
tensor_zeros = torch.zeros(2, 3)  # 2x3 tensor of zeros
print(tensor_zeros)

# Creating a tensor of ones
tensor_ones = torch.ones(2, 3)  # 2x3 tensor of ones
print(tensor_ones)

# Creating a random tensor
tensor_random = torch.randn(2, 3)  # 2x3 tensor with random values
print(tensor_random)


# ### Basic Tensor Operations
# Tensors support a variety of operations, such as element-wise arithmetic, matrix operations, and more:

# In[4]:


# Addition
sum_tensor = tensor_1 + tensor_1

# Matrix multiplication
result = torch.matmul(tensor_zeros, tensor_ones.T)

# Element-wise multiplication
elementwise_prod = tensor_1 * tensor_1


# ### Device Management
# PyTorch allows you to move tensors to different devices (CPU or GPU). For instance:

# In[6]:


# Move tensor to GPU
if torch.cuda.is_available():
    tensor_gpu = tensor_1.cuda()


# To know more about PyTorch Tensors, see the dedicated tutorial done by the PyTorch team [here](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html).

# ## Label Tensors
#
# In scientific machine learning, especially when working with **Physics-Informed Neural Networks (PINNs)**, handling tensors effectively is crucial. Often, we deal with many indices that represent physical quantities such as spatial and temporal coordinates, making it vital to ensure we use the correct indexing.
#
# For instance, in PINNs, if the wrong index is used to represent the coordinates of a physical domain, it could lead to incorrect calculations of derivatives, integrals, or residuals. This can significantly affect the accuracy and correctness of the model.
#
# ### What are Label Tensors?
#
# **Label Tensors** are a specialized type of tensor used to keep track of indices that represent specific labels. Similar to torch tensor we can perform operation, but the slicing is simplified by using indeces:

# In[7]:


# standard torch tensor
tensor = torch.randn(10, 2)

# PINA LabelTensor
label_tensor = LabelTensor(tensor, labels=["x", "y"])


# The label tensor is initialized by passing the tensor, and a set of labels. Specifically, the labels must match the following conditions:
#
# - At each dimension, the number of labels must match the size of the dimension.
# - At each dimension, the labels must be unique.
#
# For example:
#

# In[9]:


# full labels
tensor = LabelTensor(
    torch.rand((2000, 3)), {1: {"name": "space", "dof": ["a", "b", "c"]}}
)
# if you index the last column you can simply pass a list
tensor = LabelTensor(torch.rand((2000, 3)), ["a", "b", "c"])


# You can access last column labels by `.labels` attribute, or using `.full_labels` to access all labels

# In[10]:


print(f"{tensor.labels=}")
print(f"{tensor.full_labels=}")


# ### Label Tensors slicing
#
# One of the powerful features of label tensors is the ability to easily slice and extract specific parts of the tensor based on labels, just like regular PyTorch tensors but with the ease of labels.
#
# Here’s how slicing works with label tensors. Suppose we have a label tensor that contains both spatial and temporal data, and we want to slice specific parts of this data to focus on certain time intervals or spatial regions.

# In[26]:


# Create a label tensor containing spatial and temporal coordinates
x = torch.tensor([0.0, 1.0, 2.0, 3.0])  # Spatial coordinates
t = torch.tensor([0.0, 0.5, 1.0, 1.5])  # Time coordinates

# Combine x and t into a label tensor (2D tensor)
tensor = torch.stack([x, t], dim=-1)  # Shape: [4, 2]
print("Tensor:\n", tensor)

# Build the LabelTensor
label_tensor = LabelTensor(tensor, ["x", "t"])

print(f"Torch methods can be used, {label_tensor.shape=}")
print(f"also {label_tensor.requires_grad=} \n")
print(f'We can slice with labels: \n {label_tensor["x"]=}')
print(f"Similarly to: \n {label_tensor[:, 0]=}")


# You can do more complex slicing by using the extract method. For example:

# In[30]:


label_tensor = LabelTensor(
    tensor,
    {
        0: {"dof": range(4), "name": "points"},
        1: {"dof": ["x", "t"], "name": "coords"},
    },
)

print(f'Extract labels: {label_tensor.extract({"points" : [0, 2]})=}')
print(f"Similar to: {label_tensor[slice(0, 4, 2), :]=}")


# ## PyTorch Geometric Data
# PyTorch Geometric (PyG) extends PyTorch to handle graph-structured data. It provides utilities to represent graphs and perform graph-based learning tasks such as node classification, graph classification, and more.
#
# ### Graph Data Structure
# PyTorch Geometric uses a custom `Data` object to store graph data. The `Data` object contains the following attributes:
#
# - **x**: Node features (tensor of shape `[num_nodes, num_features]`)
#
# - **edge_index**: Edge indices (tensor of shape `[2, num_edges]`), representing the graph's connectivity
#
# - **edge_attr**: Edge features (optional, tensor of shape `[num_edges, num_edge_features]`)
#
# - **y**: Target labels for nodes/graphs (optional)

# In[32]:


# Node features: [2 nodes, 3 features]
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

# Edge indices: representing a graph with two edges (node 0 to node 1, node 1 to node 0)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

# Create a PyG data object
data = Data(x=x, edge_index=edge_index)

print(data)


# Once you have your graph in a Data object, you can easily perform graph-based operations using PyTorch Geometric’s built-in functions:

# In[33]:


# Accessing node features
print(data.x)  # Node features

# Accessing edge list
print(data.edge_index)  # Edge indices

# Applying Graph Convolution (Graph Neural Networks - GCN)
from torch_geometric.nn import GCNConv

# Define a simple GCN layer
conv = GCNConv(3, 2)  # 3 input features, 2 output features
out = conv(data.x, data.edge_index)
print(out)  # Output node features after applying GCN


# ## PINA Graph
#
# If you've understood Label Tensors and Data in PINA, then you're well on your way to grasping how **PINA Graph** works. Simply put, a **Graph** in PINA is a `Data` object with extra methods for handling label tensors. We highly suggest to use `Graph` instead of `Data` in PINA, expecially when using label tensors.

# In[36]:


# Node features: [2 nodes, 3 features]
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

# Edge indices: representing a graph with two edges (node 0 to node 1, node 1 to node 0)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

# Create a PINA graph object (similar to PyG)
data = Graph(x=x, edge_index=edge_index)

print(data)

# Accessing node features
print(data.x)  # Node features

# Accessing edge list
print(data.edge_index)  # Edge indices

# Applying Graph Convolution (Graph Neural Networks - GCN)
from torch_geometric.nn import GCNConv

# Define a simple GCN layer
conv = GCNConv(3, 2)  # 3 input features, 2 output features
out = conv(data.x, data.edge_index)
print(out)  # Output node features after applying GCN


# But we can also use labeltensors....

# In[40]:


# Node features: [2 nodes, 3 features]
x = LabelTensor(
    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), ["a", "b", "c"]
)

# Edge indices: representing a graph with two edges (node 0 to node 1, node 1 to node 0)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

# Create a PINA graph object (similar to PyG)
data = Graph(x=x, edge_index=edge_index)

print(data)
print(data.extract(attr="x", labels=["a"]))  # here we extract 1 feature


# In PINA Conditions, you always need to pass a list of `Graph` or `Data`, see [here]() for details. In case you are loading a PyG dataset remember to put it in this format!

# In[ ]:


from torch_geometric.datasets import QM7b

dataset = QM7b(root="./tutorial_logs").shuffle()

# save the dataset
input_ = [data for data in dataset]
input_[0]


# ## What's Next?
#
# Congratulations on completing the tutorials on the **PINA Data Structures**! You now have a solid foundation in using the different data structures within PINA, such as **Tensors**, **Label Tensors**, and **Graphs**. Here are some exciting next steps you can take to continue your learning journey:
#
# 1. **Deep Dive into Label Tensors**: Check the documentation of [`LabelTensor`](https://mathlab.github.io/PINA/_rst/label_tensor.html) to learn more about the available methods.
#
# 2. **Working with Graphs in PINA**: In PINA we implement many graph structures, e.g. `KNNGraph`, `RadiusGraph`, .... see [here](https://mathlab.github.io/PINA/_rst/_code.html#graphs-structures) for further details.
#
# 3. **...and many more!**: Consider exploring `LabelTensor` for PINNs!
#
# For more resources and tutorials, check out the [PINA Documentation](https://mathlab.github.io/PINA/).
