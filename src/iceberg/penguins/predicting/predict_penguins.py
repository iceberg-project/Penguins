"""
Wrapper for Segmentation Evaluation. 
Author: Hieu Le
License: MIT
Copyright: 2018-2019
"""
import torch
import warnings
import argparse
import os
import shutil
import time
import random
import json
import sys
import pandas as pd

from penguins.iceberg_zmq import Publisher, Subscriber
from .predict import Pipe as png_predictor
from ..options.test_options import TestOptions

warnings.filterwarnings('ignore', module='PIL')

class PenguinsPredict(object):

    def __init__(self, name, queue_in, cfg):
         
        self._name = name
        self._timings = pd.DataFrame(columns=['Image','Start','End','Seals'])
        tic = time.time()
        with open(queue_in) as fqueue:
            pub_addr_line, sub_addr_line = fqueue.readlines()

            if pub_addr_line.startswith('PUB'):
                print(pub_addr_line)
                self._in_addr_in = pub_addr_line.split()[1]
            else:
                RuntimeError('Publisher address not specified in %s' % queue_in)

            if sub_addr_line.startswith('SUB'):
                print(sub_addr_line)
                self._in_addr_out = sub_addr_line.split()[1]
            else:
                RuntimeError('Subscriber address not specified in %s' % queue_in)

        self._publisher_in = Publisher(channel=self._name, url=self._in_addr_in)
        self._subscriber_in = Subscriber(channel=self._name, url=self._in_addr_out)
        self._subscriber_in.subscribe(topic=self._name)
    
        with open(cfg) as conf:
            self._cfg = json.load(conf)
        toc = time.time()
        self._timings.loc[len(self._timings)] = ['config',tic,toc,0]

    def _connect(self):
        tic = time.time()
        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'request': 'connect',
                                                     'type': 'receiver'})
        toc = time.time()
        self._timings.loc[len(self._timings)] = ['connect',tic,toc,0]

    def _disconnect(self):
        tic = time.time()
        self._publisher_in.put(topic='request', msg={'name': self._name,
                                                     'type': 'receiver',
                                                     'request': 'disconnect'})
        toc = time.time()
        self._timings.loc[len(self._timings)] = ['disconnect',tic,toc,0]

    def _get_image(self):

        self._publisher_in.put(topic='image', msg={'request': 'dequeue',
                                                   'name': self._name})

        _, recv_message = self._subscriber_in.get()

        if recv_message[b'type'] == b'image':
            return recv_message[b'data'].decode('utf-8')

        return None

    
    def _predict_penguin(self, impath):
        opt = TestOptions().parse()
        a = png_predictor(opt)
        a.test_single_png(impath=impath)
       

    def run(self):
        #opt = TestOptions().parse()
        #a = png_predictor(opt)
        
        self._connect()

        cont = True

        while cont:
            image = self._get_image()
            sys.stdout.flush()

            if image not in ['disconnect','wait']:
                try:
                    print(image)
                    sys.stdout.flush()
                    self._predict_penguin(impath=image)
                except:
                    sys.stdout.flush()
                    print('Image not predicted', image)
                    sys.stdout.flush()
            elif image == 'wait':
                time.sleep(1)
            else:
                self._disconnect()
                cont = False
        #self._timings.to_csv(self._name + ".csv", index=False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='validates a CNN at the haul out level')
    parser.add_argument('--name', type=str)
    parser.add_argument('--queue_in', type=str)
    parser.add_argument('--config_file',type=str)
    args = parser.parse_args()

    pred = PenguinsPredict(name=args.name, queue_in=args.queue_in, cfg=args.config_file)
