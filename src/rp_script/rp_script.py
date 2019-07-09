#!/usr/bin/env python

__copyright__ = 'Copyright 2013-2014, http://radical.rutgers.edu'
__license__   = 'MIT'

import os
import sys
import time
#from __future__ import print_function
import glob
import json
import argparse
import cv2

import radical.pilot as rp
import radical.utils as ru

dh = ru.DebugHelper()

# ------------------------------------------------------------------------------
##

def args_parser():
    """
    Argument Parsing Function for the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path of the dataset')
    #parser.add_argument('resources', type=str,
    #                    help='HPC resource on which the script will run.')
    parser.add_argument('project', type=str,
                        help='The project that will be charged')
    parser.add_argument('queue', type=str,
                        help='The queue from which resources are requested.')
    parser.add_argument('runtime', type=int,
                        help='The amount of time resources are requested in' + ' minutes')
    parser.add_argument('cpus', type=int,
                        help='Number of CPU Cores required to execute')
    
    print (parser.parse_args())
    return parser.parse_args()



if __name__ == '__main__':

    args = args_parser()

    # we use a reporter class for nicer output
    report = ru.Reporter(name='radical.pilot')
    report.title('Getting Started (RP version %s)' % rp.version)


    # Create a new session. No need to try/except this: if session creation
    # fails, there is not much we can do anyways...
    session = rp.Session()

    try:

        
        # Add a Pilot Manager. Pilot managers manage one or more ComputePilots.
        pmgr = rp.PilotManager(session=session)

        # Define an [n]-core local pilot that runs for [x] minutes
        # Here we use a dict to initialize the description object
        pd_init = {'resource'      : 'xsede.bridges',
                   'runtime'       : args.runtime,  # pilot runtime (min)
                   'exit_on_error' : True,
                   'project'       : args.project,
                   'queue'         : args.queue,
                   'access_schema' : 'gsissh',
                   'cores'         : args.cpus,
		   'gpus'          : '2'
                  }
        pdesc = rp.ComputePilotDescription(pd_init)

        # Launch the pilot.
        pilot = pmgr.submit_pilots(pdesc)

        report.header('submit Image Parser unit')
        cuds_pars = list()
        # Register the ComputePilot in a UnitManager object.
        umgr = rp.UnitManager(session=session)
        umgr.add_pilots(pilot)
	
	cud_pars = rp.ComputeUnitDescription()
	cud_pars.pre_exec = '' 
        cud_pars.executable = 'python' 
        cud_pars.arguments =  ['/home/aymen/RadicalCodeTest/rp-June/img_parser.py', args.path]
        cud_pars.output_staging = {'source': 'unit:///penguins_images.json', 
                                  'target': 'client:///penguins_images.json',
                                  'action': rp.TRANSFER}
	cud_pars.gpu_processes = 0
        cud_pars.cpu_processes = 1
        cud_pars.cpu_process_type = rp.POSIX
        cud_pars.cpu_thread_type  = rp.POSIX

        cuds_pars.append(cud_pars)
        report.progress()
	umgr.submit_units(cuds_pars)
        umgr.wait_units()

        # Create a workload of ComputeUnits.

        jsonfile = open("penguins_images.json", "r")
        jsonObj = json.load(jsonfile)
        counter = 0
        n= len(jsonObj["Dataset"])# number of units to run being generated based on the number of images in the Json file 
 
        report.info('create %d unit description(s)\n\t' % n)
        cuds = list()
	report.header('submit Penguins Images units')
        for i in range(0, n):
	    img = jsonObj['Dataset'][counter]['img']

            # create a new CU description, and fill it.
            # Here we don't use dict initialization.
            cud = rp.ComputeUnitDescription()
            cud.pre_exec         =  ['source activate /pylon5/mc3bggp/aymen/anaconda3/envs/penguins22',
				     'module load psc_path/1.1',
                     		     'module load slurm/default',
                     		     'module load intel/19.3',
				     'export PYTHONPATH=/home/aymen/SummerRadical/Penguins/src:$PYTHONPATH']
            cud.executable       =  'python'
            cud.arguments        =  ['/home/aymen/SummerRadical/Penguins/src/predicting/predict.py', '--gpu_ids', 0,'--name', 'v3weakly_unetr_bs96_main_model_ignore_bad', '--epoch', 300,
                                    '--checkpoints_dir', '/home/aymen/SummerRadical/Penguins/checkpoints_dir/checkpoints_CVPR19W/', '--output', 'test', '--testset', 'GE', '--input_im', img]
            #cud.gpu_processes    = 2
            cud.cpu_processes    = 1
            cud.cpu_threads      = 1
            cud.cpu_process_type = rp.POSIX
            cud.cpu_thread_type  = rp.POSIX
            cuds.append(cud)
            report.progress()
	    counter = counter+1
        report.ok('>>ok\n')

        # Submit the previously created ComputeUnit descriptions to the
        # PilotManager. This will trigger the selected scheduler to start
        # assigning ComputeUnits to the ComputePilots.
        umgr.submit_units(cuds)

        # Wait for all compute units to reach a final state (DONE, CANCELED or FAILED).
        report.header('gather results')
        umgr.wait_units()


    except Exception as e:
        # Something unexpected happened in the pilot code above
        report.error('caught Exception: %s\n' % e)
        ru.print_exception_trace()
        raise

    except (KeyboardInterrupt, SystemExit) as e:
        # the callback called sys.exit(), and we can here catch the
        # corresponding KeyboardInterrupt exception for shutdown.  We also catch
        # SystemExit (which gets raised if the main threads exits for some other
        # reason).
        ru.print_exception_trace()
        report.warn('exit requested\n')

    finally:
        # always clean up the session, no matter if we caught an exception or
        # not.  This will kill all remaining pilots.
        report.header('finalize')
        session.close(download=True)

    report.header()


# ------------------------------------------------------------------------------

