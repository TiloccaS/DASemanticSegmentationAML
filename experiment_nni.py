import nni
import sys
import argparse
from nni.experiment import Experiment

if __name__== "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--root',
                       dest='root',
                       type=str,
                       default='../Datasets/Cityscapes')
    parse.add_argument('--root_source',
                       dest='root_source',
                       type=str,
                       default='../Datasets/GTA5')
    parse.add_argument('--root_target',
                       dest='root_target',
                       type=str,
                       default='../Datasets/Cityscapes')
    parse.add_argument('--dataset',             ## parameter added to understand whether we want to use Cityscapes or GTA
                       dest='dataset',
                       type=str, 
                       default='Cityspaces',
                       help='Select Dataset between GTAV and Cityspaces')
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall')
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='')
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=bool,
                       default=False)
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
                       default='sgd',
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

    #here we have the search for the hyper parameter that we want optimize with the respective range
    search_space = {
        'batch-size': {'_type': 'randint', '_value': [2, 12]},
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'learning_rate_D':{'_type': 'loguniform', '_value': [1e-6, 1e-3]},
        'num_epochs':{'_type': 'randint', '_value': [15, 50]},
        'lambda_adv_target1':{'_type': 'uniform', '_value': [1e-5, 1e-3]},
        'weight_decay':{'_type': 'uniform', '_value': [1e-5, 0.01]},
    }

    args = parse.parse_args()
    experiment = Experiment('local')
    experiment.config.search_space = search_space

    #This simple annealing algorithm begins by sampling from the prior 
    #but tends over time to sample from points closer and closer to the best ones observed. 
    #This algorithm is a simple variation on random search that leverages smoothness in the response surface. 
    #The annealing rate is not adaptive.
    experiment.config.tuner.name = 'Anneal' 

    #we maximize beacause we use mIoU 
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    #file train_nni.py contain the training of domain adaptation using dynamic hyperparameters
    experiment.config.trial_command = 'python train_nni.py --root ' + args.root + " --root_source " + args.root_source+\
        " --root_target " + args.root_target + " --dataset " + args.dataset + " --backbone " + args.backbone+\
        " --pretrain_path " + args.pretrain_path + " --use_conv_last " + str(args.use_conv_last) + " --epoch_start_i " + str(args.epoch_start_i)+\
        " --checkpoint_step " + str(args.checkpoint_step) + " --validation_step " + str(args.validation_step) + " --crop_height " + str(args.crop_height) + " --crop_width " + str(args.crop_width)+\
        " --num_workers " + str(args.num_workers) + " --num_classes " + str(args.num_classes) + " --cuda " + args.cuda + " --use_gpu " + str(args.use_gpu)+\
        " --save_model_path " + args.save_model_path + " --optimizer " + args.optimizer + " --loss " + args.loss + " --iter_size " + str(args.iter_size)+\
        " --domain_shift " + str(args.domain_shift) + " --domain_adaptation " + str(args.domain_adaptation) + " --momentum " + str(args.momentum) 
        
    experiment.config.trial_code_directory = "./" 
    experiment.config.tuner_gpu_indices = '0'
    experiment.config.experiment_working_directory="./nni_experiments_3" 
    experiment.config.max_trial_number = 10
    experiment.config.trial_concurrency = 1
    experiment.config.max_experiment_duration = '12h'
    
    #not always the port 8030 is available so we need to search it
    for port in range(8030,8090):
        try:
            print(port)
            experiment.run(port)
            sys.exit(0)
        except:
            pass