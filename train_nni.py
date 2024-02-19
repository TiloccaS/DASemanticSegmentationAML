import torch
from torch.utils.data import DataLoader
from model.model_stages import BiSeNet
import argparse
import numpy as np
import pandas as pd
import time
from tensorboardX import SummaryWriter
import torch
import argparse
import nni
from nni.experiment import Experiment
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset.cityscapes import CityScapes
from dataset.GTAV import GtaV
import torch.cuda.amp as amp
from model.discriminator import FCDiscriminator
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm.auto import tqdm
from train import str2bool

def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def train_DA(args, model, dataloader_val, batch_size, learning_rate, learning_rate_D, num_epochs, lambda_adv_target1, weight_decay):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    writer = SummaryWriter(comment=''.format(args.optimizer))
    scaler = amp.GradScaler()

    max_miou = 0
    step = 0
    lr= learning_rate
    lr_D1= learning_rate_D

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1=torch.nn.DataParallel(model_D1).cuda()

    source_dataset = GtaV('train', args.root_source, args.aug_type,args.crop_height,args.crop_width)
    dataloader_source = DataLoader(source_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
    target_dataset = CityScapes('train', args.root_target,args.crop_height,args.crop_width)
    dataloader_target = DataLoader(target_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
   

    optimizer = torch.optim.SGD(model.parameters(),
                          lr=lr, momentum=args.momentum, weight_decay=weight_decay)

    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=lr_D1, betas=(0.9, 0.99))


    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for epoch in range(num_epochs):

        lr=poly_lr_scheduler(optimizer,lr,epoch,max_iter=num_epochs)
        lr_D1=poly_lr_scheduler(optimizer,lr_D1,epoch,max_iter=num_epochs)
        model.train()
        model_D1.train()
      
        tq = tqdm(total=min(len(dataloader_source),len(dataloader_target))* batch_size )
        tq.set_description('epoch %d, lr_segmentation %f, lr_discriminator %f'% (epoch, lr, lr_D1))
        for i, (source_data, target_data) in enumerate(zip(dataloader_source,dataloader_target)):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False


            # train with source

            
            images, labels= source_data
            labels = labels[ :, :, :].long().cuda()
            images = images.cuda()
            optimizer.zero_grad()
            optimizer_D1.zero_grad()
          
            with amp.autocast():
                output, out16, out32 = model(images)

                loss1 = loss_func(output, labels.squeeze(1))
                loss2 = loss_func(out16, labels.squeeze(1))
                loss3 = loss_func(out32, labels.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            
            images, labels= target_data
            labels = labels[ :, :, :].long().cuda()
            images = images.cuda()
            with amp.autocast():
                output_t, out16_t, out32_t = model(images)

                D_out1=model_D1(torch.nn.functional.softmax(out32_t,dim=1)) 
                loss_adv_target1 = bce_loss(D_out1,
                                        torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())
            
                loss_D1=loss_adv_target1*lambda_adv_target1
            # proper normalization
            scaler.scale(loss_D1).backward()


            for param in model_D1.parameters():
                param.requires_grad = True  
            
            out32=out32.detach()
            out32_t=out32_t.detach() 
            

            with amp.autocast():
                D_out1=model_D1(torch.nn.functional.softmax(out32,dim=1)) 
                loss_adv_source1 = bce_loss(D_out1,
                                        torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())
                
            scaler.scale(loss_adv_source1).backward()

            with amp.autocast():

                D_out1=model_D1(torch.nn.functional.softmax(out32_t,dim=1)) 
                loss_adv_target1 = bce_loss(D_out1,
                                        torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda())

            scaler.scale(loss_adv_target1).backward()

            scaler.step(optimizer)
            scaler.step(optimizer_D1)
            scaler.update()

            loss_G = loss + loss_D1
            loss_adv = loss_adv_source1 + loss_adv_target1
            
            tq.update(batch_size)
            tq.set_postfix(loss='%.6f' % loss, loss_G='%.6f' % loss_G, loss_adv='%.6f' % loss_adv)

            step += 1
            writer.add_scalar('loss_step', loss, step)
            writer.add_scalar('loss_G', loss_G, step)
            writer.add_scalar('loss_adv', loss_adv, step)

        print('exp = {}'.format(args.save_model_path))
        
        print('iter = {0:1d}/{1}, loss_seg = {2:.3f} loss_D1 = {3:.3f}'.format(epoch, num_epochs, loss_G, loss_adv))
        tq.close()
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            print ('save model ...')
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(args.checkpoint_step) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(args.checkpoint_step) + '_D1.pth'))
            

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
            nni.report_intermediate_result(miou)
    nni.report_final_result(max_miou)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--root',
                       dest='root',
                       type=str,
                       default='../Datasets/Cityscapes',
    )
    parse.add_argument('--root_source',
                       dest='root_source',
                       type=str,
                       default='../Datasets/GTA5',
    )
    parse.add_argument('--root_target',
                       dest='root_target',
                       type=str,
                       default='../Datasets/Cityscapes',
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
                      default='pretrained_models/STDCNet813M_73.91.tar',
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
    
    args = parse.parse_args()
    ## dataset
    n_classes = args.num_classes

    root = args.root
    aug_type = args.aug_type
    #after each train you obtain the new hyperparameters
    params = nni.get_next_parameter()

    val_dataset = CityScapes(root=root,mode='val',height=args.crop_height,width=args.crop_width)
    dataloader_val = DataLoader(val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=True)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    if args.domain_adaptation:
        train_DA(args, model, dataloader_val, batch_size=params['batch-size'], learning_rate=params['learning_rate'],
                learning_rate_D=params['learning_rate_D'], num_epochs=params['num_epochs'], lambda_adv_target1=params['lambda_adv_target1'],
                weight_decay=params['weight_decay'])
        

