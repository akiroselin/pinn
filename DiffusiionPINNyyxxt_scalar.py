"""
Diffusion equation with vector input [x,y,t] and vector output [[u,v].
Weighted-Physics-Informed Neural Networks (W-PINNs)
Author: Xintan Lin
In this code we solve for the spatial temporal question: https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.html
Default example is Domain I (Square Domain, [0,1]^2), with time domain. [0,1].

"""

import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import torch.autograd as autograd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from pyDOE import lhs

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('current device: ',device)


# paramters
layers = np.array([3,64,64,2])
epoch = 5000
lr = 1e-3
x_min = -1
x_max =1
t_min = 0
t_max = 1
points_x = 200
points_t =100
N_bc= 500
N_tr = 1000

# Neural nets
class FCN(nn.Module):
    def __init__(self,layers):
        super(FCN,self).__init__()
        self.activation = nn.Tanh()
        self.loss_fucntion = nn.MSELoss(reduction = 'mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
    
    def forward(self,xyt):
        if torch.is_tensor(xyt) != True:
            xyt = torch.from_numpy(xyt)
        a = xyt.float()
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossBC(self, xy_BC, uv_BC):
        output = self.forward(xy_BC)
        loss_BC = self.loss_fucntion(output[:,[0]], uv_BC)
        return loss_BC
    
    def lossPDE(self,xy_PDE):
        xyt = xy_PDE.clone()
        xyt.requires_grad=True
        uv=self.forward(xyt)
        u = uv[:,0]
        v = uv[:,1]

        #def residual(uu,xyt):
            #u_x_y_t = autograd.grad(uu,xyt,torch.ones(uu.shape).to(device),retain_graph=True, create_graph=True)[0]
            #u_xx_yy_tt = autograd.grad(u_x_y_t,xyt,torch.ones(u_x_y_t.shape).to(device),create_graph=True)[0]
            #f = torch.exp(xyt[:,2])*(torch.sin(np.pi*xyt[:,0])*torch.sin(np.pi*xyt[:,1]) - 2 * np.pi**2*torch.sin(np.pi*xyt[:,0])*torch.sin(np.pi*xyt[:,1]))
            #u_t = u_x_y_t[:,2]
            #u_xx = u_xx_yy_tt[:,0]
            #u_yy = u_xx_yy_tt[:,1]
            #r = u_t -u_xx - u_yy + f
            #return r
        def gradients(outputs, inputs):
            return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs).to(device), create_graph=True)[0]

        def residual(uu,xyt):
            u_x_y_t = gradients(uu,xyt)
            u_x = u_x_y_t[:,0]
            u_y = u_x_y_t[:,1]
            u_t = u_x_y_t[:,2]
            u_xx  = gradients(u_x,xyt)[:,0]
            u_yy  = gradients(u_y,xyt)[:,1]
            f = torch.exp(-xyt[:,2])*(torch.sin(np.pi*xyt[:,0])*torch.sin(np.pi*xyt[:,1]) - 2* np.pi**2*torch.sin(np.pi*xyt[:,0])*torch.sin(np.pi*xyt[:,1]))
            r = u_t - u_xx - u_yy + f
            return r


        r_u = residual(u,xyt)
        r_hat = torch.zeros(xyt.shape[0]).to(device)
        return self.loss_fucntion(r_u,r_hat)
    
    def loss(self, xyt_BC,uv_BC,xyt_PDE, epoch):
        loss_bc=self.lossBC(xyt_BC,uv_BC)
        loss_pde = self.lossPDE(xyt_PDE)
        return 200*loss_bc+loss_pde
    '''
        def closure(self):
        optimizer.zero_grad()
        loss = self.loss(xyt_train_bc,uv_train_bc,xyt_train_eva)
        loss.backward()
        self.epoce += 1
        if self.epoch % 100 == 0:
            loss_test = self.lossBC(xyt_test,uv_test)
            print("Training Error:",loss.detach().cpu().numpy(),"---Testing Error:",loss2.detach().cpu().numpy())
        return loss
    '''

def solution(x,y,t):
    return torch.exp(-t)*(torch.sin(np.pi*x))*torch.sin(np.pi*y)

def plot4DS(x,y,t,u,name):
    X1,X2= torch.meshgrid(x,y)
    # Create the animation
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X1, X2,u[:,:,0] ,rstride=1, cstride=1, cmap='rainbow')
    fig.canvas.draw()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    def update(i):
        ax.clear()
        ax.set_zlim(-1, 1)
        ax.plot_surface(X1, X2,u[:,:,i] ,rstride=1, cstride=1, cmap='rainbow')
    ani = FuncAnimation(fig, update, frames=len(t), interval=1,blit=False)
    ani.save(name+'_4d_surface.gif', writer='pillow')
    #plt.show()

#  prepare real solution as test benchmark
xy = torch.linspace(x_min,x_max, points_x*2).view(-1,2)
t = torch.linspace(t_min,t_max,points_t).view(-1,1)
X,Y,T = torch.meshgrid(xy[:,0],xy[:,1],t.squeeze(1))
uv_real = solution(X,Y,T)
xyt_test = torch.hstack((X.flatten()[:,None],Y.flatten()[:,None],T.flatten()[:,None]))
uv_test = uv_real.flatten()[:,None]
low_bdry = xyt_test[0]
up_bdry = xyt_test[-1]
#plot4DS(xy[:,0],xy[:,1],t,uv_real)

# Initial boundary conditions.  (U=V on boundry)
#Left Edge: t = 0, min=<x,y=<max
left_XYT = torch.hstack((X[:,:,0].flatten()[:,None],Y[:,:,0].flatten()[:,None],T[:,:,0].flatten()[:,None]))
#print(left_XYT.shape)
#left_UV = torch.tensor([torch.sin(np.pi*left_XYT[i,:][0]*torch.sin(np.pi*left_XYT[i,:][1])) for i in range(left_XYT.shape[0])])[:,None]
left_UV = uv_real[:,:,0].flatten()[:,None]
#print(left_UV.shape)
#Bottom edge: x = min or y = min. tmin=<t=<tmax
bottom_XYT1=torch.hstack((X[0,:,:].flatten()[:,None],Y[0,:,:].flatten()[:,None],T[0,:,:].flatten()[:,None]))
bottom_XYT2=torch.hstack((X[:,0,:].flatten()[:,None],Y[:,0,:].flatten()[:,None],T[:,0,:].flatten()[:,None]))
bottom_XYT=torch.vstack((bottom_XYT1,bottom_XYT2))
bottom_UV=torch.zeros(bottom_XYT.shape[0],1)
#Top Edge: x= max or y=max, tmin=<t=<tmax
top_XYT1=torch.hstack((X[-1,:,:].flatten()[:,None],Y[-1,:,:].flatten()[:,None],T[-1,:,:].flatten()[:,None])) 
top_XYT2=torch.hstack((X[:,-1,:].flatten()[:,None],Y[:,-1,:].flatten()[:,None],T[:,-1,:].flatten()[:,None])) 
top_XYT=torch.vstack((top_XYT1,top_XYT2))
top_UV=torch.zeros(top_XYT.shape[0],1)

#prepare training data, include normal pde points, boundary conditions
XYT_train = torch.vstack([left_XYT,bottom_XYT,top_XYT])
UV_train = torch.vstack([left_UV,bottom_UV,top_UV])
print(XYT_train.shape,UV_train.shape)
idx = np.random.choice(XYT_train.shape[0], N_bc*2, replace=False)
xyt_train_bc = XYT_train[idx,:]
uv_train_bc = UV_train[idx,:]
xyt_train_eva = low_bdry + (up_bdry - low_bdry) * lhs(3, N_tr * 2)
xyt_train_eva = torch.vstack((xyt_train_eva, xyt_train_bc))
print("Original shapes for XYT and UV:", X.shape, Y.shape, T.shape, uv_real.shape)
print("Boundary shapes for the edges:",left_XYT.shape,bottom_XYT.shape,top_XYT.shape)
print("Available training data:",XYT_train.shape,UV_train.shape)
print("Final training data:",xyt_train_bc.shape,uv_train_bc.shape)
print("Total collocation points:",xyt_train_eva.shape)

#Training of neural net
xyt_train_bc = xyt_train_bc.float().to(device)
uv_train_bc = uv_train_bc.float().to(device)
xyt_train_eva =  xyt_train_eva.float().to(device)
xyt_test = xyt_test.float().to(device)
uv_test = uv_test.float().to(device)
PINN = FCN(layers)
PINN.to(device)
print(PINN)
params = list(PINN.parameters())
optimizer = torch.optim.Adam(PINN.parameters(), lr=lr,amsgrad=False)
for i in range(epoch):
    if i == 0:
        print("Training Loss-----Test Loss")
    loss = PINN.loss(xyt_train_bc,uv_train_bc,xyt_train_eva,epoch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%(epoch/10)==0:
        with torch.no_grad():
            test_loss=PINN.lossBC(xyt_test,uv_test)
        print(loss.detach().cpu().numpy(), '----', test_loss.detach().cpu().numpy())

# Plot result
uv_pinn = PINN(xyt_test)
uv_0_arr = uv_pinn[:,0].reshape(shape=[200,200,100]).detach().cpu()
uv_1_arr = uv_pinn[:,1].reshape(shape=[200,200,100]).detach().cpu()
uv_test_arr = uv_test.reshape(shape=[200,200,100]).detach().cpu()
plot4DS(X[:,0,0],Y[0,:,0],T[0,0,:],uv_0_arr,'uv1')
#plot4DS(X[:,0,0],Y[0,:,0],T[0,0,:],uv_1_arr,'uv2')
plot4DS(X[:,0,0],Y[0,:,0],T[0,0,:],uv_test_arr,'uv_solu')

























