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
from tqdm import tqdm
 

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

def train_DA(args, model):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1.train()
    model_D1.cuda(args.cpu)

    model_D2 = FCDiscriminator(num_classes=args.num_classes)
    model_D2.train()
    model_D2.cuda(args.cpu)

    model_D3 = FCDiscriminator(num_classes=args.num_classes)
    model_D3.train()
    model_D3.cuda(args.cpu)

    source_dataset = GtaV(args.mode, args.root)
    dataloader_source = DataLoader(source_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
    target_dataset = CityScapes(args.mode, args.root)
    dataloader_target = DataLoader(target_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)
    
    sourceloader_iter = enumerate(dataloader_source)
    targetloader_iter = enumerate(dataloader_target)

    optimizer = torch.optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    optimizer_D3 = torch.optim.Adam(model_D3.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D3.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for iter in range(args.num_epochs):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        loss_seg_value3 = 0
        loss_adv_target_value3 = 0
        loss_D_value3 = 0

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, iter)

        optimizer_D1.zero_grad()
        adjust_learning_rate_D(args, optimizer_D1, iter)

        optimizer_D2.zero_grad()
        adjust_learning_rate_D(args, optimizer_D2, iter)

        optimizer_D3.zero_grad()
        adjust_learning_rate_D(args, optimizer_D3, iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            for param in model_D3.parameters():
                param.requires_grad = False

            # train with source

            batch = next(iter(sourceloader_iter))
            images, labels= batch
            labels = labels[ :, :, :].long().cuda()
            images = images.cuda(args.gpu)

            output, out16, out32 = model(images)

            loss_seg1 = loss_func(output, labels.squeeze(1))
            loss_seg2 = loss_func(out16, labels.squeeze(1))
            loss_seg3 = loss_func(out32, labels.squeeze(1))
            loss = loss_seg1 + loss_seg2 + loss_seg3

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy()[0] / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy()[0] / args.iter_size
            loss_seg_value3 += loss_seg3.data.cpu().numpy()[0] / args.iter_size

            # train with target

            batch = next(iter(targetloader_iter))
            images, labels = batch
            labels = labels[ :, :, :].long().cuda()
            images = images.cuda(args.gpu)

            output, out16, out32 = model(images)

            D_out1 = model_D1(torch.nn.functional.softmax(output))
            D_out2 = model_D2(torch.nn.functional.softmax(out16))
            D_out3 = model_D3(torch.nn.functional.softmax(out32))

            loss_adv_target1 = bce_loss(D_out1,
                                        torch.FloatTensor(D_out1.data.size().fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target2 = bce_loss(D_out2,
                                        torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda(
                                            args.gpu))
            
            loss_adv_target3 = bce_loss(D_out3,
                                        torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2 + args.lambda_adv_target3 * loss_adv_target3
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy()[0] / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy()[0] / args.iter_size
            loss_adv_target_value3 += loss_adv_target3.data.cpu().numpy()[0] / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            for param in model_D3.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()
            pred3 = pred3.detach()

            D_out1 = model_D1(torch.nn.functional.softmax(pred1))
            D_out2 = model_D2(torch.nn.functional.softmax(pred2))
            D_out3 = model_D3(torch.nn.functional.softmax(pred3))

            loss_D1 = bce_loss(D_out1,
                              torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda(args.gpu))
            
            loss_D3 = bce_loss(D_out3,
                              torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D3 = loss_D3 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()[0]
            loss_D_value2 += loss_D2.data.cpu().numpy()[0]
            loss_D_value3 += loss_D3.data.cpu().numpy()[0]

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()
            pred_target3 = pred_target3.detach()

            D_out1 = model_D1(torch.nn.functional.softmax(pred_target1))
            D_out2 = model_D2(torch.nn.functional.softmax(pred_target2))
            D_out3 = model_D3(torch.nn.functional.softmax(pred_target3))
            

            loss_D1 = bce_loss(D_out1,
                              torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               torch.FloatTensor(D_out2.data.size()).fill_(target_label).cuda(args.gpu))
            
            loss_D3 = bce_loss(D_out3,
                               torch.FloatTensor(D_out2.data.size()).fill_(target_label).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D3 = loss_D3 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()
            loss_D3.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()[0]
            loss_D_value2 += loss_D2.data.cpu().numpy()[0]
            loss_D_value3 += loss_D3.data.cpu().numpy()[0]

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        optimizer_D3.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_seg3 = {4:.3f} loss_adv1 = {5:.3f}, loss_adv2 = {6:.3f} loss_adv3 = {7:.3f} loss_D1 = {8:.3f} loss_D2 = {9:.3f} loss_D3 = {10:.3f}'.format(
            iter, args.num_epochs, loss_seg_value1, loss_seg_value2, loss_seg_value3, loss_adv_target_value1, loss_adv_target_value2, loss_adv_target_value3, loss_D_value1, loss_D_value2, loss_D_value3))

        if iter >= args.checkpoint_step - 1:
            print ('save model ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if iter % args.save_pred_every == 0 and iter != 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(iter) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(iter) + '_D2.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(iter) + '_D3.pth'))

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
    #parametro aggiunto per capire se vogliamo usare cityspaces o gta
    parse.add_argument('--dataset',
                       dest='dataset',
                       type=str,
                       default='Cityspaces',
                       help='Select Dataset between GTAV and Cityspaces'
    )
    #per sceglire fra train e validation
    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
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



    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode
    root=args.root
    if args.dataset == 'GTAV':
        train_dataset = GtaV(mode,root)
        dataloader_train = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True)

        val_dataset = GtaV(root=root,mode='val')
        dataloader_val = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=True)
    else:


        train_dataset = CityScapes(mode,root)
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

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val) 

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()