#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from dataset.cityscapes import CityScapes
from dataset.GTAV import GtaV
from model.discriminator import FCDiscriminator
import torch
from torch.utils.data import DataLoader
import os
import logging 
import argparse
import numpy as np
import tensorboardX
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm.auto import tqdm
 

logger = logging.getLogger()


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


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train)* args.batch_size )
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()

            label = label[ :, :, :].long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)

                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def adjust_learning_rate(args, optimizer, iter):
    lr = poly_lr_scheduler(optimizer, args.learning_rate, iter)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, iter):
    lr = poly_lr_scheduler(optimizer, args.learning_rate_D, iter)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def train_DA(args, model, dataloader_val):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    writer = SummaryWriter(comment=''.format(args.optimizer))
    scaler = amp.GradScaler()

    max_miou = 0
    #step = 0
    lr=args.learning_rate
    lr_D1=args.learning_rate_D

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1.train()
    torch.nn.DataParallel(model_D1).cuda()

    source_dataset = GtaV('train', args.root_source)
    dataloader_source = DataLoader(source_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
    target_dataset = CityScapes('train', args.root_target)
    dataloader_target = DataLoader(target_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
   

    optimizer = torch.optim.SGD(model.parameters(),
                          lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=lr_D1, betas=(0.9, 0.99))


    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for epoch in range(args.num_epochs):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        

        lr=poly_lr_scheduler(optimizer,lr,epoch,max_iter=args.num_epochs)
        lr_D1=poly_lr_scheduler(optimizer,lr_D1,epoch,max_iter=args.num_epochs)

      
        tq = tqdm(total=min(len(dataloader_source),len(dataloader_target))* args.batch_size )
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
            
                loss_D1=loss_adv_target1*args.lambda_adv_target1
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

            loss_G=loss+loss_D1

            loss_adv=loss_adv_source1+loss_adv_target1
            
            print('exp = {}'.format(args.save_model_path))
        
            print('iter = {0:1d}/{1:8d}, loss_seg = {2:.3f} loss_D1 = {3:.3f}'.format(epoch, args.num_epochs, loss_G,  loss_adv))
            tq.update(args.batch_size)
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

        '''if iter % args.save_pred_every == 0 and iter != 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(iter) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(iter) + '_D2.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.save_model_path, 'GTA5_' + str(iter) + '_D3.pth'))'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    
    #questi sono gli argomenti da linea di comando
    #la maggior parte hanno un valore di default se volete darglielo voi dovete fare
    #python train.py --batchsize 2 (se per esempio volete modificare la batch size)
    #ho aggiunto la root che sarebbe la root del dataaset
    #il default e quella del mio pc, si dovrebbe cambiare
    
    
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
    parse.add_argument('--num_epochs',
                       type=int, default=300,
                       help='Number of epochs to train for')
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
    parse.add_argument('--batch_size',
                       type=int,
                       default=2,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='learning rate used for train')
    parse.add_argument('--learning_rate_D',
                       type=float,
                       default=0.01,
                       help='learning rate used for discriminator')
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
    parse.add_argument('--weight_decay',
                       type=float,
                       default=5e-4,
                       help='Regularisation parameter for L2-loss')
    parse.add_argument('--lambda_adv_target1',
                       type=float,
                       default=0.0002,
                       help='lambda_adv for adversarial training')
    parse.add_argument('--lambda_adv_target2',
                       type=float,
                       default=0.001,
                       help='lambda_adv for adversarial training')
    parse.add_argument('--lambda_adv_target3',
                       type=float,
                       default=0.001,
                       help='lambda_adv for adversarial training')
    parse.add_argument('--lambda_seg',
                       type=float,
                       default=0.1,
                       help='lambda_seg')
    


    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    root=args.root
    if args.dataset == 'GTAV':
        train_dataset = GtaV('train',root)
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)

        val_dataset = GtaV(root=root, mode='val')
        dataloader_val = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=True)
    else:


        train_dataset = CityScapes('train', root)
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)

        val_dataset = CityScapes(root=root,mode='val')
        dataloader_val = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=True)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        print("sto usando la GPU")

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    if args.domain_adaptation:
        print(True)
        train_DA(args, model, dataloader_val)

    if not args.domain_shift:
        ## train loop
        train(args, model, optimizer, dataloader_train, dataloader_val)
        
    # final test
    val(args, model, dataloader_val)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()