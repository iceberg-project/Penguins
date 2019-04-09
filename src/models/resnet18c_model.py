"""
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from resnet18 import resnet18

class Resnet18Cmodel(BaseModel):
    def name(self):
        return 'Resnet18 for Classification Model'
    def eval(self):
        self.netG.eval()
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        
        # load/define networks
        self.netG = resnet18(out_nc= 1,gpu_ids= self.gpu_ids)
        
        self.criterionMSE= torch.nn.MSELoss()        
        bce_logit = torch.nn.BCEWithLogitsLoss().cuda()
        self.criterion = bce_logit
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
        else:
            self.load_network(self.netG, 'G', opt.which_epoch)
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')
        

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input = Variable(input_A)
        self.GT  = Variable(input_B)
        bGT,tmp = torch.max(self.GT.view(self.GT.shape[0],-1),dim=1)
        self.binary_GT = Variable(bGT).view(-1,1)
        self.binary_GT[self.binary_GT==-1]=0


    def forward(self):
        self.output = self.netG(self.input)
        batchsize = self.output.shape[0]
        nc = self.output.shape[1]
        self.output = torch.mean(self.output.view([batchsize,nc,-1]),dim=2)



    def backward_G(self):
        self.loss_G = self.criterion(self.output, self.binary_GT)*100 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_Loss', self.loss_G.detach().cpu())])
    def get_prediction_tensor(self,input_A):
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input = Variable(input_A)
        self.output = self.netG(self.input)
        raw = self.output.data.cpu().float().numpy()
        return OrderedDict([('input',util.tensor2im(self.input.data)),('output',util.tensor2im(self.output.data)),('raw_out',raw)])

    def get_prediction(self,input):
        self.input = Variable(input['A'].cuda(self.gpu_ids[0], async=True))
        self.output = self.netG(self.input)
        raw = self.output.data.cpu().float().numpy()
        raw = np.transpose(raw,(0,2,3,1))
        return OrderedDict([('input',util.tensor2im(self.input.data)),('output',util.tensor2im(self.output.data)),('raw_out',raw)])



    def get_current_visuals(self):
       	self.visual_names = ['input','GT'] 
        nim = getattr(self,self.visual_names[0]).shape[0]
        visual_ret = OrderedDict()
        all =[]
        for i in range(0,min(nim-1,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            row=np.hstack(tuple(row))
            row = util.drawtext2im(row,'GT: ' +str(self.binary_GT[i].data.cpu().float().numpy()),pos=(10,10),size=25)
            row = util.drawtext2im(row,'Pr: ' +str(self.output[i].data.cpu().float().numpy()),pos=(266,10),size=25)
            if self.binary_GT[i]==0 and self.output[i] > 0:
                row = util.addborder(row,fill=(200,0,0),border=8)
            if self.binary_GT[i]==1 and self.output[i] < 0:
                row = util.addborder(row,fill=(200,0,0),border=8)
            all.append(row)
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
