import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn.utils.weight_norm as weight_norm
import time
import pickle

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
        
        self.Nx = 257
        self.Nt = 101
        self.layers = layers
        self.device = device
        
        t = torch.linspace(0.0, 1.0, self.Nt)
        x = torch.linspace(-1.0, 1.0, self.Nx)
        xx,tt = torch.meshgrid(x,t)
        
        self.X_ref = torch.cat([xx.reshape(-1,1),tt.reshape(-1,1)],dim=1).to(self.device); 
        
        self.X_ic = torch.cat([xx[:,[0]],tt[:,[0]]],dim=1).to(self.device)
        self.u_ic = (xx[:,[0]]**2*torch.cos(torch.pi*xx[:,[0]])).to(self.device)
        
        self.X_lbc = torch.cat([xx[[0]],tt[[0]]],dim=0).T.to(self.device); self.X_lbc.requires_grad = True
        self.X_ubc = torch.cat([xx[[-1]],tt[[-1]]],dim=0).T.to(self.device); self.X_ubc.requires_grad = True

        self.log = {'losses':[], 'losses_b':[], 'losses_i':[], 'losses_f':[], 'losses_s':[], 'time':[]}
        
        self.min_loss = 1
        self.model = Net(layers, self.X_ref.cpu().detach().numpy(), self.device).to(self.device)

    def Msei(self):
        u = self.model(self.X_ic)
        msei = F.mse_loss(u, self.u_ic)
        return msei
    
    def Mseb(self):
        X = self.X_lbc
        u_lbc = self.model(X)
        u_xt = fwd_gradients(u_lbc, X)
        u_lbc_x = u_xt[:,0:1]
        
        X = self.X_ubc
        u_ubc = self.model(X)
        u_xt = fwd_gradients(u_ubc, X)
        u_ubc_x = u_xt[:,0:1]
        
        mseb = F.mse_loss(u_ubc, u_lbc) + F.mse_loss(u_ubc_x, u_lbc_x) 
        return mseb
    
    def TimeStepping(self):

        X = self.X
        u = self.model(X)

        self.U0 = u.detach()
    
    def Msef(self):

        X = self.X
        u = self.model(X)
        u_xt = fwd_gradients(u, X)
        u_x = u_xt[:,0:1]
        u_t = u_xt[:,1:2]
        u_xx = fwd_gradients(u_x, X)[:,0:1]

        f = u_t - 0.0001*u_xx + 5*u**3 - 5*u
        dt = 0.3
        msef = 1/dt**2*((u - self.U0 + dt*f)**2).mean()
        return msef
    
    def Mses(self):
        X = self.X_ref
        u = self.model(X)
        mses = torch.norm(u-self.u_ref) / torch.norm(self.u_ref)
        return mses

    def Loss(self):
        msei = self.Msei()
        mseb = self.Mseb()
        msef = self.Msef()
        loss = 10 * msei + 1 * mseb + 1 * msef

        return loss, msei, mseb, msef

    def ResidualPoint(self):
        self.X = torch.rand((20000,2), device=self.device)
        self.X[:,0] = self.X[:,0] * 2 - 1; self.X.requires_grad = True
        
    def train(self, epoch):

        if len(self.log['time']) == 0:
            t1 = time.time()
        else:
            t1 = time.time() - self.log['time'][-1]
        

        for i in range(epoch):
            def closure():
                self.optimizer.zero_grad()
                self.loss, self.loss_i, self.loss_b, self.loss_f = self.Loss()
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
            self.log['losses_i'].append(self.loss_i.item())
            self.log['losses_s'].append(self.loss_s.item())
            self.log['time'].append(t2-t1)
            
            if (i % np.clip(int(epoch/1000),1,1000) == 0) or (i == epoch - 1):
                print(f'{i}|{epoch} loss={self.loss.item()} PDE loss={self.loss_f.item()} error={self.loss_s.item()}')
                
if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if 1 else "cpu")
    
    layers = [2, 128, 128, 128, 128, 1]
    
    nn = TSONN(layers, device)
    
    u_ref = np.loadtxt('u_ref.dat').T
    nn.u_ref = torch.tensor(u_ref.reshape(-1,1)).to(device)
    nn.train(1000)
    
    t2 = time.time()
    print(t2 - t1)

    # %%
    plt.figure()
    plt.plot(nn.log['time'], np.log10(nn.log['losses']), label='loss')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_s']), label='Reletive L2 error')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_f']), label='loss_f')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_b']), label='loss_b')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_i']), label='loss_i')

    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    # %%
    XX = nn.X_ref[:,0].cpu().detach().numpy().reshape(nn.Nx,nn.Nt)
    TT = nn.X_ref[:,1].cpu().detach().numpy().reshape(nn.Nx,nn.Nt)
    u_pred = nn.model(nn.X_ref).cpu().detach().numpy().reshape(nn.Nx,nn.Nt)
    
    
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Reference')
    plt.tight_layout()
    
    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'iTSONN')
    plt.tight_layout()
    
    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()
    # %%
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(XX[:,0], u_ref[:,0], color='blue')
    plt.plot(XX[:,0], u_pred[:,0], '--', color='red')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.title('$t = 0$')
    plt.tight_layout()
    
    plt.subplot(1, 3, 2)
    plt.plot(XX[:,0], u_ref[:,50], color='blue')
    plt.plot(XX[:,0], u_pred[:,50], '--', color='red')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.title('$t = 0.5$')
    plt.tight_layout()
    
    plt.subplot(1, 3, 3)
    plt.plot(XX[:,0], u_ref[:,-1], color='blue')
    plt.plot(XX[:,0], u_pred[:,-1], '--', color='red')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.title('$t = 1.0$')
    plt.tight_layout()
    plt.show()