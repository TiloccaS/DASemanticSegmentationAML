import nni
from nni.experiment import Experiment
import os
import argparse
import sys

if __name__== "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--root',
                       dest='root',
                       type=str,
                       default='/mnt/d/Salvatore/Reboot/Universita/VANNO/AdvancedMachineLearning/ProjectAML/Cityscapes/Cityspaces',
    )
    parse.add_argument('--root_source',
                       dest='root_source',
                       type=str,
                       default='/mnt/d/Salvatore/Reboot/Universita/VANNO/AdvancedMachineLearning/ProjectAML/Cityscapes/Cityspaces',
    )
    parse.add_argument('--root_target',
                       dest='root_target',
                       type=str,
                       default='/mnt/d/Salvatore/Reboot/Universita/VANNO/AdvancedMachineLearning/ProjectAML/Cityscapes/Cityspaces',
    )
    #parametro aggiunto per capire se vogliamo usare cityspaces o gta
    parse.add_argument('--dataset',
                       dest='dataset',
                       type=str,
                       default='Cityspaces',
                       help='Select Dataset between GTAV and Cityspaces'
    )
 
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )

    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=1,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to modelwork')
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to modelwork')
    parse.add_argument('--num_workers',
                       type=int,
                       default=0,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default=None,
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    parse.add_argument('--iter_size',
                       type=int,
                       default=1,
                       help='Accumulate gradients for ITER_SIZE iterations')
    parse.add_argument('--domain_shift',
                       type=bool,
                       default=False,
                       help='To test domain shift from GTAV to Cityscapes')
    parse.add_argument('--domain_adaptation',
                       type=bool,
                       default=False,
                       help='To train domain adaptation from GTAV to Cityscapes')
    parse.add_argument('--momentum',
                       type=float,
                       default=0.9,
                       help='Momentum component of the optimiser')
    parse.add_argument('--aug_type',
                       type=str,
                       default=None,
                       help='type of Data Augmentation to apply')

    search_space = {
        'batch-size': {'_type': 'randint', '_value': [2, 16]},
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'learning_rate_D':{'_type': 'loguniform', '_value': [1e-6, 1e-3]},
        'num_epochs':{'_type': 'randint', '_value': [25, 50]},
        'lambda_adv_target1':{'_type': 'uniform', '_value': [1e-5, 1e-5]},
        'weight_decay':{'_type': 'uniform', '_value': [1e-5, 0.01]},
    }
    args = parse.parse_args()
    experiment = Experiment('local')
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'Anneal'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.trial_command = 'python train_nni.py --root ' +args.root+" --root_source "+args.root_source+\
        " --root_target "+args.root_target +" --dataset "+args.dataset +" --backbone "+args.backbone+\
        " --pretrain_path "+args.pretrain_path+" --use_conv_last "+str(args.use_conv_last)+" --epoch_start_i "+str(args.epoch_start_i)+\
        " --checkpoint_step "+str(args.checkpoint_step)+" --validation_step "+str(args.validation_step)+" --crop_height "+str(args.crop_height)+" --crop_width "+str(args.crop_width)+\
        " --num_workers "+str(args.num_workers)+" --num_classes "+str(args.num_classes)+" --cuda "+args.cuda
    experiment.config.trial_code_directory = "./" #'./sbonito'
    #experiment.config.trial_gpu_number=1
    #experiment.config.use_active_gpu=True

    experiment.config.experiment_working_directory="./nni_experiments" #'./sbonito/nni-experiments'

    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 1

    #experiment.config.max_experiment_duration = '5m'

    for port in range(8020,8090):
        try:
            experiment.run(port)
            sys.exit(0)
        except:
            pass