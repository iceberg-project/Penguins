"""
Calling the BaseModel, DataLoader, and Visualizer wrappers for training the models.
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""

import sys
sys.path.insert(0, "./../")
sys.path.insert(0,"/nfs/bigbox/hieule/GAN/code/ICEBERG_Penguins/src/")
sys.path.insert(0,"/nfs/bigbox/hieule/GAN/code/ICEBERG_Penguins/src/model")
from models import create_model
import os,gc,datetime,time,socket
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
import visdom
import time
from data import CreateDataLoader
from util.visualizer import Visualizer
hostname=socket.gethostname()


opt = TrainOptions().parse()

visualizer = Visualizer(opt)
vis2 = visdom.Visdom(server = opt.display_host,port = 7007)
total_steps = 0
data_loader = CreateDataLoader(opt)
#dataset = TifDataset(opt)
dataset = data_loader.load_data()
print('dataset size: ' + str(len(dataset)))
dataset_size = len(data_loader)
model = create_model(opt)
print(opt)
for epoch in range(opt.epoch_count,opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    epoch_iter=0
    
    print('epoch:%d'%(epoch))

    for i,data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()

        vis2.text(str(opt.display_port)+'  ' + hostname+str(opt.gpu_ids)+'  EPOCH:'+ '   '+str(epoch) + '  ' +opt.name
                ,win=opt.display_port,opts={'title': str(opt.display_port)+':' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')})
        if i% 5 ==0:
            visualizer.display_current_results(model.get_current_visuals(), epoch, False)
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, errors,time.time()-iter_start_time)
            visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    if epoch % opt.save_epoch_freq ==0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
        model.save('latest')
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
