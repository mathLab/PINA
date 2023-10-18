# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:48:10 2023

@author: nokmo
"""

import scipy.io as spio

def loadmat(filename, ConvertoToDict = True):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    if ConvertoToDict:
        return _check_keys(data)
    else:
        return data

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _getData(FileName):
    '''
    Load time data and FEM model from .mat file. A class should
    be written to access data and plot routines more conveniently
    '''
    ImportedData = loadmat(FileName)
    NodalSol = ImportedData["DataMRT"] # extract time data (nodal solutions)
    FEMmodel = ImportedData["DataMRT"]["Parent"] # extract FEM model

    FEMmodel['Elems'] = FEMmodel['Elems']-1 # imported from MATLAB, indices begin with 1


    return NodalSol, FEMmodel


# Get time data and FEMmodel
NodalSol, FEMmodel = _getData("doc_DataFEMheat_results.DataMRT.mat")

import pickle
pickle.dump(FEMmodel, open("FEMmodel", "wb"))
pickle.dump(NodalSol, open("NodalSol", "wb"))

# %% Plot mesh
import matplotlib.pyplot as plt

# Get vertices and faces
v = FEMmodel['Nodes']
f = FEMmodel['Elems']
# import numpy as np
# NODE = np.hstack((FEMmodel["Nodes"],np.zeros((FEMmodel["Nodes"].shape[0],1))))

# display the mesh
def plot_fem_mesh(NODES, ELEMS):
    for ELEM in ELEMS:
        x = [NODES[ELEM[i],0] for i in range(len(ELEM))]
        y = [NODES[ELEM[i],1] for i in range(len(ELEM))]
        plt.fill(x, y, edgecolor=[.7 , .7 , .7], fill=False)

fig = plt.figure()
plot_fem_mesh(v, f)
plt.xlabel('x in m'), plt.ylabel('y in m')
plt.title('%s: mesh' % FEMmodel["Name"])

# %% Plot FE matrices
fig = plt.figure()
plt.spy(FEMmodel["System"]["K"],marker='.',markersize=4)
plt.title('%s: FE matrices' % FEMmodel["Name"])
# "System" contains the FE matrices, e.g. K is the stiffness matrix, M the mass matrix
# see https://de.mathworks.com/help/pde/ug/assemblefematrices.html, section "Algorithms"


# %% Plot temperature history
import numpy as np

# Select node to be plotted (here by index)
idx = (5,59,57) # Node IDs-1

t = NodalSol['XData'][:] # time
y = NodalSol['YData'][:,idx] # temperature

# Plot temperature
fig = plt.figure()
plt.plot(t,y)
plt.xlabel('Time in s'), plt.ylabel('Temperature in K'), plt.grid(True)
plt.legend(['Node ' + str(i) for i in idx ])
plt.title('%s: nodal solution' % FEMmodel["Name"])

# Plot source term
q = FEMmodel["System"]["u"][:,:-1] # heat source q(t,x) evaluated at quadrature points
idx = np.any(q, axis=1)

fig = plt.figure()
plt.plot(t,np.transpose(q[idx,:]))
plt.xlabel('Time in s'), plt.ylabel('Heat source in W/m³'), plt.grid(True)
plt.title('%s: heat source at quadrature points' % FEMmodel["Name"])
plt.xlim(0,2)
plt.show()


# %% Plot temperature distribution
import matplotlib.tri as tri

# Select time instant (index) to plot
idx = -5
y = NodalSol["YData"][idx,:]
t = NodalSol["XData"][idx]

# create an unstructured triangular grid instance
triangulation = tri.Triangulation(v[:,0], v[:,1], f)
# this will not work with nodes in 3D space, will modify later

# plot the contours
fig = plt.figure()
plt.tricontourf(triangulation, y, cmap='RdYlBu_r')
# in MATLAB this would be: patch('Vertices',v,'Faces',f,'FaceColor','interp','FaceVertexCData,u)

# show
hcb = plt.colorbar()
hcb.set_label('Temperature in K')
plt.axis('equal')
plt.title('%s: temperature at t=%.2f s' % (FEMmodel["Name"], t))
plt.xlabel('x in m'), plt.ylabel('y in m')
plt.show()

# %% Plot heat source term

# Get inhomogeneous term
idx = 100
t = NodalSol["XData"][idx]
q = FEMmodel["System"]["F"][:,idx]

# plot the contours
fig = plt.figure()
plt.tricontourf(triangulation, q, cmap='viridis' )

# show
hcb = plt.colorbar()
hcb.set_label('Nodal heat source in W/m³')
plt.axis('equal')
plt.title('%s: heat source at t=%.2f s' % (FEMmodel["Name"], t))
plt.xlabel('x in m'), plt.ylabel('y in m')
plt.show()



