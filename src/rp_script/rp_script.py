#!/usr/bin/env python
# pylint: disable=invalid-name
'''
Radical Pilot Code for Penguins-ICEBERG
'''
import json
import argparse
import radical.pilot as rp
import radical.utils as ru

__copyright__ = 'Copyright 2013-2014, http://radical.rutgers.edu'
__license__ = 'MIT'

dh = ru.DebugHelper()

# ------------------------------------------------------------------------------
##


def args_parser():
    """
    Argument Parsing Function for the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path of the dataset')
    parser.add_argument('resources', type=str,
                        help='HPC resource on which the script will run.')
    parser.add_argument('project', type=str,
                        help='The project that will be charged')
    parser.add_argument('queue', type=str,
                        help='The queue from which resources are requested.')
    parser.add_argument('runtime', type=int,
                        help='The amount of time resources are requested in'
                        + ' minutes')
    parser.add_argument('cpus', type=int,
                        help='Number of CPU Cores required to execute')
    parser.add_argument('gpus', type=int,
                        help='Numbe of GPUs  required')
    parser.add_argument('input_pth', type=str,
                        help='Path of the source code where it existed')
    parser.add_argument('device', type=int,
                        help='CUDA device to export')
    parser.add_argument('user_env', type=str, nargs='?',
                        help='path to the User conda or virtual enviroment')
    print parser.parse_args()
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
        pd_init = {
            'resource': args.resources,
            'runtime': args.runtime,  # pilot runtime (min)
            'exit_on_error': True,
            'project': args.project,
            'queue': args.queue,
            'access_schema': 'gsissh',
            'cores': args.cpus,
            'gpus': args.gpus
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
        cud_pars.pre_exec = ['module load anaconda2', 'which python',
                             'conda info --envs']
        cud_pars.executable = 'python'
        cud_pars.arguments = ['img_parser.py', args.path]
        cud_pars.input_staging = {'source': 'client:///img_parser.py',
                                  'target': 'unit:///img_parser.py',
                                  'action': rp.TRANSFER}

        cud_pars.output_staging = {'source': 'unit:///penguins_images.json',
                                   'target': 'client:///penguins_images.json',
                                   'action': rp.TRANSFER}
        cud_pars.gpu_processes = 0
        cud_pars.cpu_processes = 1
        cud_pars.cpu_process_type = rp.POSIX
        cud_pars.cpu_thread_type = rp.POSIX

        cuds_pars.append(cud_pars)
        report.progress()
        umgr.submit_units(cuds_pars)
        umgr.wait_units()

        # Create a workload of ComputeUnits.

        jsonfile = open("penguins_images.json", "r")
        jsonObj = json.load(jsonfile)
        counter = 0
        # number of units to run being generated based on
        # the number of images in the Json file
        n = len(jsonObj["Dataset"])
        report.info('create %d unit description(s)\n\t' % n)
        cuds = list()
        report.header('submit Penguins Images units')

        gpu_id = args.device
        for i in range(0, n):
            img = jsonObj['Dataset'][counter]['img']

            # create a new CU description, and fill it.
            # Here we don't use dict initialization.
            cud = rp.ComputeUnitDescription()
            cud.pre_exec = [
                'module load anaconda2',
                'which python',
                'module list',
                'source activate %sanaconda3/envs/penguins2' % args.user_env,
                'which python',
                'export PYTHONPATH=%s/Penguins/'
                + 'src:$PYTHONPATH' % args.input_pth,
                'export PYTHONPATH=%sanaconda3/envs/'
                + 'penguins2/lib/python2.7/site-packages:'
                + '$PYTHONPATH' % args.user_env,
                'export CUDA_VISIBLE_DEVICES=%d' % gpu_id]
            cud.executable = 'python'
            cud.arguments = ['%sPenguins/src/predicting/'
                             + 'predict.py' % args.input_pth,
                             '--gpu_ids', 0, '--name',
                             'v3weakly_unetr_bs96_main_model_ignore_bad',
                             '--epoch', 300, '--checkpoints_dir',
                             '%sPenguins/checkpoints_dir/'
                             + 'checkpoints_CVPR19W/' % args.input_pth,
                             '--output', 'test', '--testset', 'GE',
                             '--input_im', img]
            cud.input_staging = {
                'source': '%sPenguins/checkpoints_dir'
                          + '/checkpoints_CVPR19W/'
                          + 'v3weakly_unetr_bs96_main_model_ignore'
                          + '_bad/300_net_G.pth' % args.input_pth,
                'target': 'unit:///Penguins/checkpoints_dir'
                          + '/checkpoints_CVPR19W/v3weakly_unetr_bs96_'
                          + 'main_model_ignore_bad/300_net_G.pth',
                'action': rp.COPY}
            cud.gpu_processes = 1
            cud.cpu_processes = 1
            cud.cpu_threads = 1
            cud.cpu_process_type = rp.POSIX
            cud.cpu_thread_type = rp.POSIX
            gpu_id = gpu_id ^ 1
            cuds.append(cud)
            report.progress()
            counter = counter + 1
        report.ok('>>ok\n')

        # Submit the previously created ComputeUnit descriptions to the
        # PilotManager. This will trigger the selected scheduler to start
        # assigning ComputeUnits to the ComputePilots.
        umgr.submit_units(cuds)

        # Wait for all compute units to reach a
        # final state (DONE, CANCELED or FAILED).
        report.header('gather results')
        umgr.wait_units()

    except Exception as e:
        # Something unexpected happened in the pilot code above
        report.error('caught Exception: %s\n' % e)
        ru.print_exception_trace()
        raise

    except (KeyboardInterrupt, SystemExit) as e:
        ru.print_exception_trace()
        report.warn('exit requested\n')

    finally:
        # always clean up the session, no matter if we caught an exception or
        # not.  This will kill all remaining pilots.
        report.header('finalize')
        session.close(download=True)

    report.header()


# ------------------------------------------------------------------------------
