import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class UnetModel(BaseModel):
    def name(self):
        return 'UnetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 1, 64,
                                      'unet_256', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.criterionL1 = torch.nn.L1Loss()        
        self.criterionMSE= torch.nn.MSELoss()        
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
#        print('---------- Networks initialized -------------')
#        networks.print_network(self.netG)
#        print('-----------------------------------------------')
        

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
        self.input = Variable(input_A)
        self.GT  = Variable(input_B)
        self.cmask = Variable(input_C) #only compute loss on these pixels
    def forward(self):
        self.output = self.netG(self.input)



    def backward_G(self):
        #self.loss_G = self.criterionL1(self.output, self.GT) 
        #self.loss_G = self.criterionMSE(self.output, self.GT) 
       # indx = self.cmask ==1
        
        #self.loss_G = self.criterionMSE(self.output[indx],self.GT[indx])
        self.loss_G = self.criterionMSE(self.output.mul(self.cmask),self.GT.mul(self.cmask)) 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_Loss', self.loss_G.data[0]),('X',0)
                            ])
    def get_prediction_tensor(self,input_A):
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input = Variable(input_A)
        self.output = self.netG(self.input)
        raw = self.output[0].data.cpu().float().numpy()
        raw = np.transpose(raw,(1,2,0))
        return OrderedDict([('input',util.tensor2im(self.input.data)),('output',util.tensor2im(self.output.data)),('raw_out',raw)])

    def get_prediction(self,input):
        self.input = Variable(input['A'].cuda(self.gpu_ids[0], async=True))
        self.output = self.netG(self.input)
        raw = self.output.data.cpu().float().numpy()
        raw = np.transpose(raw,(0,2,3,1))
        return OrderedDict([('input',util.tensor2im(self.input.data)),('output',util.tensor2im(self.output.data)),('raw_out',raw)])




    def get_current_visuals(self):
        
        allll = []
        for i in range(0,5):
            inp = util.tensor2im(self.input.data[i:i+1,:,:,:])
            out = util.tensor2im(self.output.data[i:i+1,:,:,:])
            GT = util.tensor2im(self.GT.data[i:i+1,:,:,:])
            alll = np.hstack((inp,GT,out))
            allll.append(alll)
        X =  np.vstack((allll[0],allll[1]))
        for i in range(2,5):
            X = np.vstack((X,allll[i]))
            
        
        return OrderedDict([(self.opt.name+' input-GT-out'+str(self.gpu_ids[0]),X)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
