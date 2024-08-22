import torch
import torch.nn as nn
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from datasets import ISPRSSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from tqdm import tqdm
from utils.visualizer import Visualizer
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils import loss
from utils.memory import memory_sampling_balanced
from utils.contrastive_learning import class_contrastive_learning
from fightingcv_attention.attention.DANet import DAModule  
from fightingcv_attention.attention.PSA import PSA 
from fightingcv_attention.attention.ECAAttention import ECAAttention 
from fightingcv_attention.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention 

torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='', help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='ISPRS', choices=['voc','ade','ISPRS'], help='')
    parser.add_argument("--num_classes", type=int, default=None, help="")
    parser.add_argument("--cutout", action="store_true", default=False)
    parser.add_argument("--cutmix", action="store_true", default=False)
    parser.add_argument("--mixup", action="store_true", default=False)
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101', 
                                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                        'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                        'deeplabv3_mobilenet_v2_bubbliiiing', 'deeplabv3plus_mobilenet_v2_bubbliiiing',
                                        'deeplabv3_mobilenet_v2', 'deeplabv3plus_mobilenet_v2',
                                        'deeplabv3_mobilenet_v3_small', 'deeplabv3plus_mobilenet_v3_small',
                                        'deeplabv3_mobilenet_v3_large', 'deeplabv3plus_mobilenet_v3_large',
                                        'deeplabv3_mobilenet_v3_large_test', 'deeplabv3plus_mobilenet_v3_large_test',
                                        'deeplabv3_berniwal_swintransformer_swin_t', 'deeplabv3plus_berniwal_swintransformer_swin_t',
                                        'deeplabv3_berniwal_swintransformer_swin_s', 'deeplabv3plus_berniwal_swintransformer_swin_s',
                                        'deeplabv3_berniwal_swintransformer_swin_b', 'deeplabv3plus_berniwal_swintransformer_swin_b',
                                        'deeplabv3_berniwal_swintransformer_swin_l', 'deeplabv3plus_berniwal_swintransformer_swin_l',
                                        'deeplabv3_microsoft_swintransformer_swin_t', 'deeplabv3plus_microsoft_swintransformer_swin_t',
                                        'deeplabv3_microsoft_swintransformer_swin_s', 'deeplabv3plus_microsoft_swintransformer_swin_s',
                                        'deeplabv3_microsoft_swintransformer_swin_b', 'deeplabv3plus_microsoft_swintransformer_swin_b',
                                        'deeplabv3_microsoft_swintransformer_swin_l', 'deeplabv3plus_microsoft_swintransformer_swin_l'], help='')
    parser.add_argument("--separable_conv", action='store_true', default=False, help="")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50, help="")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'], help="")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False, help='')
    parser.add_argument("--batch_size", type=int, default=32, help='')
    parser.add_argument("--val_batch_size", type=int, default=1, help='')
    parser.add_argument("--crop_size", type=int, default=513)   
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--loss_type", type=str, default='bce_loss')
    parser.add_argument("--KD_loss_type", type=str, default='KLDiv_loss')
    parser.add_argument("--use_KD_layer_weight", action='store_true', default=False, help='')
    parser.add_argument("--KD_outlogits", action='store_true', default=False, help='')
    parser.add_argument("--use_KD_class_weight", action='store_true', default=False,  help='')     
    parser.add_argument('--lamb', type=float, default=0.9, help="")               
    parser.add_argument("--gpu_id", type=str, default='0',  help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='')
    parser.add_argument("--random_seed", type=int, default=1,  help="")
    parser.add_argument("--print_interval", type=int, default=10, help="")
    parser.add_argument("--val_interval", type=int, default=200,  help="")
    parser.add_argument("--pseudo", action='store_true', help="")
    parser.add_argument("--pseudo_thresh", type=float, default=0.5)
    parser.add_argument("--task", type=str, default='4-1', help="")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="")
    parser.add_argument("--mem_size", type=int, default=0, help="")
    parser.add_argument("--freeze", action='store_true', help="")
    parser.add_argument("--bn_freeze", action='store_true', help="")
    parser.add_argument("--w_transfer", action='store_true', help="")
    parser.add_argument("--unknown", action='store_true', help="")
    parser.add_argument("--cont_learn", action='store_true', help="")
    parser.add_argument("--rho", type=float, default=1.0, help='')
    parser.add_argument("--uncertainty_pseudo", action='store_true', help='')
    return parser

def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    if opts.dataset == 'ISPRS':
        dataset = ISPRSSegmentation
    else:
        raise NotImplementedError
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    return dataset_dict

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: 
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"IL Task : {opts.task}")
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet_v2_bubbliiiing': network.deeplabv3_mobilenet_v2_bubbliiiing,
        'deeplabv3plus_mobilenet_v2_bubbliiiing': network.deeplabv3plus_mobilenet_v2_bubbliiiing,
        'deeplabv3_mobilenet_v2': network.deeplabv3_mobilenet_v2,
        'deeplabv3plus_mobilenet_v2': network.deeplabv3plus_mobilenet_v2,
        'deeplabv3_mobilenet_v3_small': network.deeplabv3_mobilenet_v3_small,
        'deeplabv3plus_mobilenet_v3_small': network.deeplabv3plus_mobilenet_v3_small,
        'deeplabv3_mobilenet_v3_large': network.deeplabv3_mobilenet_v3_large,
        'deeplabv3plus_mobilenet_v3_large': network.deeplabv3plus_mobilenet_v3_large,
        'deeplabv3_mobilenet_v3_large_test': network.deeplabv3_mobilenet_v3_large_test,
        'deeplabv3plus_mobilenet_v3_large_test': network.deeplabv3plus_mobilenet_v3_large_test,
        'deeplabv3_berniwal_swintransformer_swin_t': network.deeplabv3_berniwal_swintransformer_swin_t,
        'deeplabv3_berniwal_swintransformer_swin_s': network.deeplabv3_berniwal_swintransformer_swin_s,
        'deeplabv3_berniwal_swintransformer_swin_b': network.deeplabv3_berniwal_swintransformer_swin_b,
        'deeplabv3_berniwal_swintransformer_swin_l': network.deeplabv3_berniwal_swintransformer_swin_l,        
        'deeplabv3plus_berniwal_swintransformer_swin_t': network.deeplabv3plus_berniwal_swintransformer_swin_t,
        'deeplabv3plus_berniwal_swintransformer_swin_s': network.deeplabv3plus_berniwal_swintransformer_swin_s,
        'deeplabv3plus_berniwal_swintransformer_swin_b': network.deeplabv3plus_berniwal_swintransformer_swin_b,
        'deeplabv3plus_berniwal_swintransformer_swin_l': network.deeplabv3plus_berniwal_swintransformer_swin_l,       
        'deeplabv3_microsoft_swintransformer_swin_t': network.deeplabv3_microsoft_swintransformer_swin_t,
        'deeplabv3_microsoft_swintransformer_swin_s': network.deeplabv3_microsoft_swintransformer_swin_s,
        'deeplabv3_microsoft_swintransformer_swin_b': network.deeplabv3_microsoft_swintransformer_swin_b,
        'deeplabv3_microsoft_swintransformer_swin_l': network.deeplabv3_microsoft_swintransformer_swin_l,
        'deeplabv3plus_microsoft_swintransformer_swin_t': network.deeplabv3plus_microsoft_swintransformer_swin_t,
        'deeplabv3plus_microsoft_swintransformer_swin_s': network.deeplabv3plus_microsoft_swintransformer_swin_s,
        'deeplabv3plus_microsoft_swintransformer_swin_b': network.deeplabv3plus_microsoft_swintransformer_swin_b,
        'deeplabv3plus_microsoft_swintransformer_swin_l': network.deeplabv3plus_microsoft_swintransformer_swin_l,
    }
    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=bn_freeze)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    if opts.curr_step > 0:
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride, bn_freeze=bn_freeze)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)
    if opts.freeze and opts.curr_step > 0:
        for param in model_prev.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.head[-1].parameters(): 
            param.requires_grad = True
        training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': opts.lr}]     
        if opts.unknown:
            for param in model.classifier.head[0].parameters(): 
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
            for param in model.classifier.head[1].parameters(): 
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})
    else:
        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]   
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=opts.lr, 
                                momentum=0.9, 
                                weight_decay=opts.weight_decay,
                                nesterov=True)
    def save_ckpt(path):
        torch.save({"model_state": model.module.state_dict(), "optimizer_state": optimizer.state_dict(),"best_score": best_score,}, path)
    utils.mkdir('checkpoints')
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0  
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d.pth"
    if opts.curr_step > 0: 
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step-1)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True) 
        if opts.unknown and opts.w_transfer:
            curr_head_num = len(model.classifier.head) - 1
            checkpoint[f"classifier.head.{curr_head_num}.0.weight"] = checkpoint["classifier.head.0.0.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.weight"] = checkpoint["classifier.head.0.1.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.bias"] = checkpoint["classifier.head.0.1.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_mean"] = checkpoint["classifier.head.0.1.running_mean"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_var"] = checkpoint["classifier.head.0.1.running_var"]
            last_conv_weight = model.state_dict()[f"classifier.head.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.head.{curr_head_num}.3.bias"]
            for i in range(opts.num_classes[-1]):
                last_conv_weight[i] = checkpoint["classifier.head.0.3.weight"]
                last_conv_bias[i] = checkpoint["classifier.head.0.3.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.head.{curr_head_num}.3.bias"] = last_conv_bias
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint 
    model = nn.DataParallel(model) 
    mode = model.to(device)
    mode.train()
    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()
        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)
    if not opts.crop_val:
        opts.val_batch_size = 1
    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // 100)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, reduction='mean')
    else:
        raise NotImplementedError
    if opts.KD_loss_type == 'KLDiv_loss':
        criterion_KD = torch.nn.KLDivLoss(size_average=True, reduce=True)
    elif opts.KD_loss_type == 'KD_loss':
        criterion_KD = utils.loss.KnowledgeDistillationLoss(alpha=2.0)
    else:
        raise NotImplementedError
    if opts.prototype_matching:
        criterion_PM = torch.nn.KLDivLoss(size_average=True, reduce=True)
    if opts.use_KD_layer_weight:
        KD_layer_weight = {'l1':0.5,'l2':1.0, 'l3':2.0, 'out':1.0}
    if opts.feature_decoupling:
        criterion_DEC = utils.DCL(temperature=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    avg_KD_loss_ret = AverageMeter()
    avg_CL_loss = AverageMeter()
    KD_loss_ret = torch.tensor(0.)
    KD_loss_ret_l1 = torch.tensor(0.)
    KD_loss_ret_l2 = torch.tensor(0.)
    KD_loss_ret_l3 = torch.tensor(0.)
    KD_loss_ret_out = torch.tensor(0.)
    KD_loss_outlogits = torch.tensor(0.)
    CL_loss = torch.tensor(0.)
    SI_loss = torch.tensor(0.)
    SS_loss = torch.tensor(0.)
    decouple_loss = torch.tensor(0.)
    if opts.prototype_matching:
        SS_p_feat = [] 
        SI_p_feat = [] 
    model.train()
    save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()
        try:
            images, labels, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, _ = train_iter.next()
            cur_epochs += 1
            avg_loss.reset()
            avg_time.reset()
            avg_KD_loss_ret.reset()
            avg_CL_loss.reset()   
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)

        if opts.cutout and random.random()>0.5:
            cutout_func = et.Cutout(1, 16) 
            for i in range(opts.batch_size):
                images[i] = cutout_func(images[i])
                labels[i] = cutout_func(labels[i])
                sal_maps[i] = cutout_func(sal_maps[i])
        with torch.cuda.amp.autocast(enabled=opts.amp):
            ret_features, outputs = model(images) 
            if opts.feature_decoupling:
                psa = SequentialPolarizedSelfAttention(channel=256)
                psa = psa.to(device)
                decouple_feat = ret_features['feature_out']
                decouple_feat  = decouple_feat.type(torch.float16) 
                decouple_feat = psa(decouple_feat).to(device) 
            if opts.prototype_matching:
                SS_p_feat.append(ret_features['feature_l2'])
                SI_p_feat.append(ret_features['feature_out'])
                SS_p_feat = SS_p_feat - bg_p_feat
            if model_prev is not None and opts.curr_step>0:
                with torch.no_grad():
                    ret_features_prev, outputs_prev = model_prev(images)
                if opts.feature_decoupling:
                    channel_num = ret_features_prev['feature_out'].size(2)
                    SS_channel_num = round(opts.rho * channel_num)
                    SI_channel_num = channel_num - SS_channel_num
                    sim_list = []
                    decouple_loss = torch.tensor(0.).to(device)
                    for curr_channel in range(channel_num):
                        sim = criterion_KD(ret_features_prev['feature_out'][:,curr_channel,:,:], ret_features['feature_out'][:,curr_channel,:,:]).to(device)
                        sim_list.append(sim)
                        decouple_loss += torch.as_tensor(sim / channel_num)
                if opts.cont_learn:
                    CL_loss = class_contrastive_learning(outputs, ret_features['feature_out'], outputs_prev, ret_features_prev['feature_out'], num_classes=5)
                lamb = math.pow(opts.lamb, (cur_epochs/opts.train_epoch))
                if opts.use_KD_layer_weight:
                    KD_loss_ret_l1 = KD_layer_weight['l1'] * lamb * criterion_KD(ret_features['feature_l1'], ret_features_prev['feature_l1'])
                    KD_loss_ret_l2 = KD_layer_weight['l2'] * lamb * criterion_KD(ret_features['feature_l2'], ret_features_prev['feature_l2'])
                    KD_loss_ret_l3 = KD_layer_weight['l3'] * lamb * criterion_KD(ret_features['feature_l3'], ret_features_prev['feature_l3'])
                elif opts.prototype_matching:
                    KD_loss_PM = criterion_KD(ret_features['feature_out'], ret_features_prev['feature_out']) 
                else:
                    KD_loss_ret_l1 = criterion_KD(ret_features['feature_l1'], ret_features_prev['feature_l1'])
                    KD_loss_ret_l2 = criterion_KD(ret_features['feature_l2'], ret_features_prev['feature_l2'])
                    KD_loss_ret_l3 = criterion_KD(ret_features['feature_l3'], ret_features_prev['feature_l3'])                   
                if opts.feature_decoupling:
                    DEC_loss = criterion_DEC(ret_features['feature_out'], ret_features_prev['feature_out'])
                    KD_loss_ret = KD_loss_ret_l1 + KD_loss_ret_l2 + KD_loss_ret_l3 + DEC_loss + decouple_loss
                else:
                    KD_loss_ret = KD_loss_ret_l1 + KD_loss_ret_l2 + KD_loss_ret_l3 + KD_loss_ret_out       
            if opts.pseudo and opts.curr_step > 0:
                with torch.no_grad():
                    ret_features_prev, outputs_prev = model_prev(images)
                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                pred_scores, pred_labels = torch.max(pred_prob, dim=1) 
                if opts.uncertainty_pseudo:
                    prev_num_classes = pred_prob.shape[1]
                    value_scores, indices_scores = pred_prob.topk(2,dim=1, largest=True, sorted=True) 
                    u_map = torch.zeros_like(value_scores)
                    u_map = value_scores[:, 0, :, :] - value_scores[:, 1, :, :]
                    ranking_map = torch.mul((torch.ones_like(u_map)-u_map), pred_labels)
                    pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh) & (ranking_map > opts.pseudo_thresh), pred_labels, labels)
                else: 
                    pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh), pred_labels, labels)
                loss = criterion(outputs, pseudo_labels)
            else:
                loss = criterion(outputs, labels)
        if opts.cont_learn:
            CL_loss.requires_grad_(True)
            scaler.scale(CL_loss.to(device)).backward(retain_graph=True)
            avg_CL_loss.update(CL_loss.item())
        if model_prev is not None and opts.curr_step > 0:
            KD_loss_ret.requires_grad_(True)
            scaler.scale(KD_loss_ret).backward(retain_graph=True)
            avg_KD_loss_ret.update(KD_loss_ret.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        end_time = time.time()
    if opts.curr_step > 0:
        best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))
        class_iou = list(test_score['IoU'].values())
        first_cls = len(get_tasks(opts.dataset, opts.task, 0))

if __name__ == '__main__':
    opts = get_argparser().parse_args()
    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset, opts.task))
    for step in range(start_step, total_step):
        opts.curr_step = step
        main(opts)
        
