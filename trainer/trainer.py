import torch
from torch import optim
import numpy as np
from models import *
from utils import *


class HybridDeepSVDD:
    def __init__(self, dataset, lr, wd, ms, ptep, ep, bs, dv, pt, aod, ald, dod, dnl, axl, gd, cs, vp, train, valid):
        self.dataset = dataset
        self.lr = lr
        self.wd = wd
        self.ms = ms
        self.ptep = ptep
        self.ep = ep
        self.bs = bs
        self.dv = dv
        self.pt = pt
        self.aod = aod
        self.ald = ald
        self.dod = dod
        self.dnl = dnl
        self.axl = axl
        self.gd = gd  
        self.cs = cs
        self.vp = vp        
        self.train = train
        self.valid = valid
        self.validt = torch.tensor(valid).float()      
        self.c = None
        self.cb = None
        self.net = None
        self.netb = None
        

    def pretrain(self):
        input_dim = self.aod
        ae_latent_dim = self.ald
        dsvdd_latent_dim = self.dod
        dsvdd_num_layers = self.dnl
        ae_extra_layers = self.axl
        granularity_diff = self.gd
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = AE(input_dim, dsvdd_latent_dim, ae_latent_dim, dsvdd_num_layers, ae_extra_layers).to(self.dv)
        aeb = AE(input_dim, dsvdd_latent_dim, ae_latent_dim, dsvdd_num_layers - granularity_diff, ae_extra_layers).to(self.dv)    
        optimizer = optim.Adam(ae.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.ms, gamma=0.1)
        optimizerb = optim.Adam(aeb.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        schedulerb = optim.lr_scheduler.MultiStepLR(optimizerb, 
                    milestones=self.ms, gamma=0.1)        
        for epoch in range(self.ptep):
            ae.train()
            aeb.train()           
            running_loss = 0            
            running_lossb = 0
            iteration = 0
            for i in range(0, len(self.train), self.bs):
                iteration = iteration + 1
                b = i + self.bs
                x = torch.tensor(self.train[i:b,:]).float().to(self.dv)
                optimizer.zero_grad()
                optimizerb.zero_grad()                
                x_hat = ae(x)
                x_hatb = aeb(x)                
                train_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                train_lossb = torch.mean(torch.sum((x_hatb - x) ** 2, dim=tuple(range(1, x_hatb.dim()))))                
                train_loss.backward()
                optimizer.step()
                train_lossb.backward()
                optimizerb.step()                  
                running_loss += train_loss.item() 
                running_lossb += train_lossb.item()                 
            scheduler.step()
            schedulerb.step()            
            if self.vp:
                with torch.no_grad():
                    ae.eval()
                    aeb.eval()                    
                    pred_val = chunkfeed(ae, self.valid, self.cs, self.aod)   
                    pred_valb = chunkfeed(aeb, self.valid, self.cs, self.aod)                     
                    val_loss = torch.mean(torch.sum((pred_val - self.validt) ** 2, dim=tuple(range(1, pred_val.dim()))))
                    val_lossb = torch.mean(torch.sum((pred_valb - self.validt) ** 2, dim=tuple(range(1, pred_valb.dim()))))                         
            loss = running_loss / iteration 
            lossb = running_lossb / iteration                
            if self.vp:        
                print("A-epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, self.ptep, loss, val_loss))
                print("B-epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, self.ptep, lossb, val_lossb))
            else:
                print("A-epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, self.ptep, loss))
                print("B-epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, self.ptep, lossb))
        self.save_weights_for_DeepSVDD(self.dataset, ae, input_dim, dsvdd_latent_dim, dsvdd_num_layers, "a")
        self.save_weights_for_DeepSVDD(self.dataset, aeb, input_dim, dsvdd_latent_dim, dsvdd_num_layers-granularity_diff, "b")  
                        
        
    def save_weights_for_DeepSVDD(self, dataset, model, input_dim, latent_dim, num_layers, letter):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        net = DSVDD(input_dim, latent_dim, num_layers).to(self.dv)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        c = self.set_c(net)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters_' + dataset +  '_' + str(input_dim) + '_' + str(latent_dim) + "_" + str(num_layers) + letter + '.pth')
                    

    def set_c(self, model, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for i in range(0, len(self.train), self.bs):
                b = i + self.bs
                x = torch.tensor(self.train[i:b,:]).float().to(self.dv)
                z = model(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def maintrain(self):
        input_dim = self.aod
        latent_dim = self.dod
        num_layers = self.dnl
        granularity_diff = self.gd
        """Training the Deep SVDD model"""
        net = DSVDD(input_dim, latent_dim, num_layers).to(self.dv)
        netb = DSVDD(input_dim, latent_dim, num_layers-granularity_diff).to(self.dv)        
        if self.pt:
            state_dict = torch.load('weights/pretrained_parameters_' + self.dataset +  '_' + str(input_dim) + '_' + str(latent_dim) + "_" + str(num_layers) + "a" + '.pth')
            state_dictb = torch.load('weights/pretrained_parameters_' + self.dataset +  '_' + str(input_dim) + '_' + str(latent_dim) + "_" + str(num_layers-granularity_diff) + "b" + '.pth')
            net.load_state_dict(state_dict['net_dict'])
            netb.load_state_dict(state_dictb['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.dv)
            cb = torch.Tensor(state_dictb['center']).to(self.dv)
            np.savetxt("center.csv", c.detach().numpy(), delimiter=",")
            np.savetxt("centerb.csv", cb.detach().numpy(), delimiter=",")            
        else:
            net.apply(weights_init_normal)
            c = torch.randn(ld).to(self.dv)
            np.savetxt("center.csv", c.detach().numpy(), delimiter=",")
        
        optimizer = optim.Adam(net.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.ms, gamma=0.1)
        optimizerb = optim.Adam(netb.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        schedulerb = optim.lr_scheduler.MultiStepLR(optimizerb, 
                    milestones=self.ms, gamma=0.1)
                    
        for epoch in range(self.ep):
            net.train()        
            running_loss = 0
            iteration = 0
            for i in range(0, len(self.train), self.bs):
                iteration = iteration + 1
                b = i + self.bs
                x = torch.tensor(self.train[i:b,:]).float().to(self.dv)

                optimizer.zero_grad()
                optimizerb.zero_grad()
                z = net(x)
                zb = netb(x)                
                train_loss = torch.mean(torch.sum((z - c) ** 2, dim=1)) + torch.mean(torch.sum((zb - cb) ** 2, dim=1))
                train_loss.backward()
                optimizer.step()
                optimizerb.step()
                running_loss += train_loss.item()
            scheduler.step()
            schedulerb.step()           
            if self.vp:
                with torch.no_grad():
                    net.eval()
                    netb.eval()
                    pred_val = chunkfeed(net, self.valid, self.cs, self.dod)
                    pred_valb = chunkfeed(netb, self.valid, self.cs, self.dod)                    
                    val_loss = torch.mean(torch.sum((pred_val - c) ** 2, dim=1)) + torch.mean(torch.sum((pred_valb - cb) ** 2, dim=1))
            loss = running_loss / iteration
            if self.vp:        
                print("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, self.ep, loss, val_loss))
            else:
                print("epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, self.ep, loss))
        self.net = net
        self.netb = netb
        self.c = c.detach().numpy()
        self.cb = cb.detach().numpy()
          
