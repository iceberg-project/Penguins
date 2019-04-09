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
from PIL import ImageOps,Image

class UnetRModel(BaseModel):
    def name(self):
        return 'UnetModelWithRegressionLoss'
    def eval(self):
        self.netG.eval()
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 1, 64,
                                      'unet_256', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.criterionL1 = torch.nn.L1Loss()        
        self.criterion= torch.nn.MSELoss()        
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
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input = Variable(input_A)
        self.GT  = Variable(input_B)
        self.count=  Variable(input['counts'].cuda(self.gpu_ids[0], async=True))
        if 'isweak' in input:
            self.isweak = Variable(input['isweak'].cuda(self.gpu_ids[0], async=True))
        else:
            self.isweak = Variable(torch.zeros(self.count.shape).cuda(self.gpu_ids[0],async=True))
 
 
        if 'isfixed' in input:
            self.isfixed = Variable(input['isfixed'].cuda(self.gpu_ids[0], async=True))
        else:
            self.isfixed = Variable(torch.zeros(self.count.shape).cuda(self.gpu_ids[0],async=True))


    def forward(self):
        self.output = self.netG(self.input)



    def backward_G(self):
        #segmentation loss

        num = self.output.shape[0]

        if not hasattr(self.opt,'lambda_segmentation'):
            self.opt.lambda_segmentation=100
        self.loss_G_segmentation = self.criterion(self.output[self.isweak==0,:,:,:], self.GT[self.isweak==0,:,:,:]) * self.opt.lambda_segmentation 

        
        if not hasattr(self.opt,'lambda_regression'):
            self.opt.lambda_regression=500
        #regression loss:
        self.loss_G_regression = self.criterion(self.output.view(num,-1).mean(dim =1)/2+0.5,self.count) * self.opt.lambda_regression
        self.loss_G = self.loss_G_segmentation + self.loss_G_regression
        self.loss_G.backward()

    def get_current_visuals(self):
       	self.visual_names = ['input','output','GT'] 
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
            row=tuple(row)
            row = np.hstack(row)
            if self.isweak[i] == 0:
                row = ImageOps.crop(Image.fromarray(row),border =8)
                row = ImageOps.expand(row,border=8,fill=(0,200,0))
                row = np.asarray(row)
            elif self.isfixed[i] == 1:
                row = ImageOps.crop(Image.fromarray(row),border =4)
                row = ImageOps.expand(row,border=4,fill=(0,0,200))
                row = np.asarray(row)


            all.append(row)
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        
        return OrderedDict([('G_Loss', self.loss_G.detach().cpu()),
                            ('G_Regression',self.loss_G_regression.detach().cpu()),
                            ('G_segmentation',self.loss_G_segmentation.detach().cpu())])
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





    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
