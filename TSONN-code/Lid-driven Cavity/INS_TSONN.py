import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn.utils.weight_norm as weight_norm
import time
import pickle
import copy

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph= True)[0]
    return G

class Net(torch.nn.Module):
    def __init__(self, layer_dim, X, device):
        super().__init__()

        self.X_mean = torch.from_numpy(X.mean(0, keepdims=True)).float()
        self.X_std = torch.from_numpy(X.std(0, keepdims=True)).float()
        self.X_mean = self.X_mean.to(device)
        self.X_std = self.X_std.to(device)
        
        self.num_layers = len(layer_dim)
        temp = []
        for l in range(1, self.num_layers):
            temp.append(weight_norm(torch.nn.Linear(layer_dim[l-1], layer_dim[l]), dim = 0))
            torch.nn.init.normal_(temp[l-1].weight)
        self.layers = torch.nn.ModuleList(temp)
        
    def forward(self, x):
        x = ((x - self.X_mean) / self.X_std) # z-score norm
        for i in range(0, self.num_layers-1):
            x = self.layers[i](x)
            if i < self.num_layers-2:
                x = torch.tanh(x)
        return x
    
class TSONN():

    def __init__(self, layers, device):
        self.layers = layers
        self.device = device

        Nx = 501
        Ny = 501
        
        x = torch.linspace(0, 1, Nx)
        y = torch.linspace(0, 1, Ny)
        [xx, yy] = torch.meshgrid([x, y])
        xx, yy = xx.T.to(self.device), yy.T.to(self.device)
        
        self.X_wall = torch.cat([torch.cat([xx[1:-1,[0]],xx[1:-1,[-1]],xx[[0]].T]), \
                                    torch.cat([yy[1:-1,[0]],yy[1:-1,[-1]],yy[[0]].T])], dim = 1)
        self.X_lid = torch.cat([xx[[-1],1:-1].T, yy[[-1],1:-1].T], dim = 1)

        self.X_ref = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim = 1)
        
        self.Nx = self.Ny = 501

        self.min_loss = 1
        self.log = {'losses':[], 'losses_b':[], 'losses_i':[], 'losses_f':[], 'losses_s':[], 'time':[]}
        
        self.model = Net(self.layers, self.X_ref.cpu().detach().numpy(), self.device).to(self.device)

    def Mseb(self):

        pred_lid = self.model(self.X_lid)
        mseb1 = ((pred_lid[:,0]-1)**2 + (pred_lid[:,1])**2).mean()

        pred_wall = self.model(self.X_wall)
        mseb2 = ((pred_wall[:,0])**2 + (pred_wall[:,1])**2).mean()

        return mseb1 + mseb2
    
    def TimeStepping(self):
        X = self.X
        pred = self.model(X)
        u = pred[:,0:1];  v = pred[:,1:2];   p = pred[:,2:3]; 
        
        self.U0 = torch.cat([u,v,p]).detach()

    def Msef(self):
        X = self.X
        pred = self.model(X)
        u = pred[:,0:1];  v = pred[:,1:2];   p = pred[:,2:3]; 
              
        u_xy = fwd_gradients(u, X)
        v_xy = fwd_gradients(v, X)
        p_xy = fwd_gradients(p, X)
        u_x = u_xy[:,0:1]; u_y = u_xy[:,1:2]
        v_x = v_xy[:,0:1]; v_y = v_xy[:,1:2]
        p_x = p_xy[:,0:1]; p_y = p_xy[:,1:2]
        
        u_xx = fwd_gradients(u_x, X)[:,0:1]
        u_yy = fwd_gradients(u_y, X)[:,1:2]
        v_xx = fwd_gradients(v_x, X)[:,0:1]
        v_yy = fwd_gradients(v_y, X)[:,1:2]

        #非守恒
        res_rho = u_x + v_y
        res_u = u*u_x + v*u_y + p_x - 1/self.Re*(u_xx + u_yy)
        res_v = u*v_x + v*v_y + p_y - 1/self.Re*(v_xx + v_yy)
        
        msef = (res_u**2 + res_v**2 + res_rho**2).mean() 
        
        U1 = torch.cat([u,v,p])
        R1 = torch.cat([res_u,res_v,res_rho])
        
        dtau = 0.5
        msef = 1/dtau**2*((U1 - self.U0 + dtau*R1)**2).mean()
        return msef
    
    def Mses(self):
        pred = self.model(self.X_ref)
        u = pred[:,0:1]; v = pred[:,1:2];
        V = torch.sqrt(u**2 + v**2).detach().reshape(self.Nx,self.Ny)
        mses = torch.norm((V - self.V_ref).reshape(-1),2) / torch.norm((self.V_ref).reshape(-1),2)
        return mses
    
    def Loss(self):
        mseb = self.Mseb()
        msef = self.Msef()
        
        loss = 1 * mseb + msef
        return loss, mseb, msef
    
    def ResidualPoint(self):
        self.X = torch.rand((20000,2), device=self.device, requires_grad=True)
        
    def train(self, epoch):

        if len(self.log['time']) == 0:
            t1 = time.time()
        else:
            t1 = time.time() - self.log['time'][-1]

        for i in range(epoch):
            def closure():
                self.optimizer.zero_grad()
                self.loss, self.loss_b, self.loss_f = self.Loss()
                self.loss.backward()
                return self.loss

            # Because the loss function in TSONN is constantly changing,
            # the optimizer must be reinitialized in each outer iteration!
            # Thus, the resiudal points can be updated in each outer iteration.
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=300)
            
            self.ResidualPoint() 
            self.TimeStepping()
            self.optimizer.step(closure)
            
            self.loss_s = self.Mses()
            
            # exceptional handling
            if (self.loss != self.loss) or ((i>1) and (self.loss.item() > 3*self.log['losses'][-1])):
                if i == 0:
                    self.model = Net(self.layers, self.X_ref.cpu().detach().numpy(), self.device).to(self.device)
                    continue
                else:
                    self.model = torch.load('model_temp.pth')
                    print('load model')
                    
                    #restart optimizer with new collection point
                    self.ResidualPoint() 
                    self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=300)
                    continue
            if i % 3 == 0:
                torch.save(self.model,'model_temp.pth')

            
            t2 = time.time()
            self.log['losses'].append(self.loss.item())
            self.log['losses_f'].append(self.loss_f.item())
            self.log['losses_b'].append(self.loss_b.item())
            self.log['losses_s'].append(self.loss_s.item())
            self.log['time'].append(t2-t1)
            
            if (i % np.clip(int(epoch/1000),1,1000) == 0) or (i == epoch - 1):
                print(f'{i}|{epoch} loss={self.loss.item()} PDE loss={self.loss_f.item()} error={self.loss_s.item()}')
 
if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if 1 else "cpu")
    
    layers = [2, 128, 128, 128, 128, 128, 3]
    
    nn = TSONN(layers, device)
    
    Data = np.loadtxt('Re100_V_ref.dat')
    Data = Data.reshape(1001, 1001, 3)
    V_ref = Data[::2,::2,2].T/30;  #Nondimensionalization! The original data is dimensional
    nn.V_ref = torch.tensor(V_ref.T).to(device)
    
    nn.Re = 100
    nn.train(300)

    t2 = time.time()
    print('total time: ', t2-t1)
    # %%
    plt.figure()
    plt.plot(nn.log['time'], np.log10(nn.log['losses']), label='loss')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_s']), label='Reletive L2 error')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_f']), label='loss_f')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_b']), label='loss_b')

    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    
    
    plt.figure()
    X = nn.X_ref[:,[0]].reshape(nn.Nx,nn.Ny).cpu().detach()
    Y = nn.X_ref[:,[1]].reshape(nn.Nx,nn.Ny).cpu().detach()
    pred = nn.model(nn.X_ref).cpu().detach()
    V = torch.sqrt(pred[:,0]**2 + pred[:,1]**2).cpu().detach().reshape(nn.Nx,nn.Ny)
    plt.contourf(X,Y,V)
    plt.colorbar()