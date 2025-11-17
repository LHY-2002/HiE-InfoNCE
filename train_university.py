import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel

from my_util import LossLogger
from FSRA.datasets.U1652Dataset import *
from FSRA.datasets.make_dataloader import *
from datetime import datetime
import pytz
import argparse
from sample4geo.Multi_InfoLoss import Multi_InfoNCE
from shutil import copyfile


@dataclass
class Configuration:  # 适用于参数多而且复杂，参数之间有联系或需要条件判断，还可以显示参数类型
    
    # Model
    #model: str = 'convnext_base.fb_in22k_ft_in1k_384'  # 带类型提示和默认值的参数构造方法
    model: str = 'vit_base_patch16_384.augreg_in21k_ft_in1k'

    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 1
    epochs: int = 120
    batch_size: int = 64                # keep in mind real_batch_size = 2 * batch_size
    sample_num: int = 1
    verbose: bool = True
    # gpu_ids: tuple = (0,1,2,3)        # GPU ids for training
    gpu_ids: tuple = (0,)
    save_weights: bool = True           # weather save weights
    multi: float = 1.0

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 5          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int (评估样本数，-1=全部) 无用参数

    # Optimizer 
    clip_grad = 100.                     # None | float  梯度裁剪的阈值
    decay_exclue_bias: bool = False      # 优化器使用权重衰减
    grad_checkpointing: bool = True      # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1         # 标签平滑因子
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    logit_scale_lr: float = 0.001
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 5               # 预热轮数
    lr_end: float = 0.0001               # only for "polynomial"
    
    # Dataset
    dataset: str = 'U1652-D2S'            # 'U1652-D2S' | 'U1652-S2D'
    #dataset: str = 'U1652-S2D'
    data_folder: str = r"E:\University-Release\train"
    
    # Augment Images
    prob_flip: float = 0.5                # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./university"
    weights_save_path: str = "./university"
    
    # Eval before training
    zero_shot: bool = False             # 评估未训练模型的初始性能
    
    # Checkpoint to start from
    checkpoint_start = None             # resume train
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True        # 自动寻找最优卷积算法，提升速度
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False



#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = r'E:\University-Release\train\satellite'
    config.gallery_folder_train = r'E:\University-Release\train\drone'
    config.query_folder_test = r'E:\University-Release\test\query_drone'
    config.gallery_folder_test = r'E:\University-Release\test\gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = r'E:\University-Release\train\satellite'
    config.gallery_folder_train = r'E:\University-Release\train\drone'
    config.query_folder_test = r'E:\University-Release\test\query_satellite'
    config.gallery_folder_test = r'E:\University-Release\test\gallery_drone'

#-----------------------------------------------------------------------------#
# 脚本运行传入参数
#-----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description='Run Bash')
parser.add_argument('--epochs', default=120, type=int, help='the epoch of training')
parser.add_argument('--multi', default=1.0, type=float, help='the multi of scheduler')
parser.add_argument('--batch_size', default=30, type=int, help='the batch_size of data')
parser.add_argument('--sample_num', default=3, type=int, help='multi sampling')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--scheduler', default='cosine', type=str, help='the scheduler of lr')
parser.add_argument('--warmup_epochs', default=5, type=int, help='the warmup_epochs for training')
opt = parser.parse_args()

config.epochs = opt.epochs
config.multi = opt.multi
config.batch_size = opt.batch_size
config.sample_num = opt.sample_num
config.lr = opt.lr
if opt.scheduler == 'cosine' or opt.scheduler == 'constant' or opt.scheduler == 'polynomial':
    config.scheduler = opt.scheduler
else:
    raise(ValueError("{} is not a valid scheduler".format(opt.scheduler)))
config.warmup_epochs = opt.warmup_epochs


if __name__ == '__main__':

    # 创建模型保存路径
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz).strftime("%Y_%m_%d_%H%M")
    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   current_time)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #创建模型权重保存文件夹
    config.weights_save_path = os.path.join(model_path,'weights')
    if not os.path.exists(config.weights_save_path):
        os.makedirs(config.weights_save_path)

    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))  # 将当前的训练程序复制到模型保存目录

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'train_log.txt'))  # 创建双输出日志（控制台+log.txt）所有print()输出将同时显示在控制台和日志文件

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    ########################################
    # 训练说明
    print("\n{}[*训练说明*]{}".format(30 * "-", 30 * "-"))
    print(f"epoch = {config.epochs}，"
          f"multi = {config.multi}，"
          f"batchsize = {config.batch_size}，"
          f"sample_num = {config.sample_num}，"
          f"使用{config.scheduler} scheduler，"
          f"lr = {config.lr}，"
          f"warmup_epochs = {config.warmup_epochs}，")

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
                          
    data_config = model.get_config()  # 获取模型的配置信息
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    else:
        config.device = torch.device(f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    data_transforms = get_transforms(img_size, mean=mean, std=std)
    val_transforms = data_transforms['val']
    train_sat_transforms = data_transforms['satellite']
    train_drone_transforms = data_transforms['drone']
                                                                                                                                 
    # Train
    train_dataloader, class_names, dataset_sizes = get_train_dataloader(config, data_transforms)
    print('dataset_sizes:', "(satellite:{}，drone:{})".format(dataset_sizes['satellite'], dataset_sizes['drone']))
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),  # 以query的label为准
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    multi_info_loss = Multi_InfoNCE(loss_function=loss_fn,
                                    device=config.device,
                                    sample_num=config.sample_num)

    if config.mixed_precision:
        scaler = GradScaler('cuda', init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    """
    将模型参数分为两组：
    应用权重衰减的参数（权重参数）
    不应用权重衰减的参数（偏置和层归一化参数）
    为每组参数设置不同的权重衰减系数
    """
    factor_1 = round((14 / len(train_dataloader)) * 1, 2)
    factor_2 = round((14 / len(train_dataloader)) * 2, 2)
    print("\nTemperature coefficient scaling factor:")
    print("factor_1: {}  factor_2: {}".format(factor_1,factor_2))
    
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
    
        no_decay = ["bias", "LayerNorm.bias"]
    
        # 先排除 logit_scale 参数
        exclude_params = ["logit_scale1", "logit_scale2"]
    
        decay_params = [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
            and not any(ep in n for ep in exclude_params)
        ]
        no_decay_params = [
            p for n, p in param_optimizer
            if any(nd in n for nd in no_decay)
            and not any(ep in n for ep in exclude_params)
        ]
    
        optimizer_parameters = [
            {"params": decay_params, "weight_decay": 0.01, "lr": config.lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": config.lr},
            {"params": [model.logit_scale1], "weight_decay": 0.0, "lr": config.logit_scale_lr * factor_1},
            {"params": [model.logit_scale2], "weight_decay": 0.0, "lr": config.logit_scale_lr * factor_2},
        ]
    
        optimizer = torch.optim.AdamW(optimizer_parameters)

    else:
        params = [
            {'params': model.model.parameters(), 'lr': config.lr},
            {'params': model.logit_scale1, 'lr': config.logit_scale_lr * factor_1},
            {'params': model.logit_scale2, 'lr': config.logit_scale_lr * factor_2},
        ]
        optimizer = torch.optim.AdamW(params)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = int(len(train_dataloader) * config.epochs * config.multi)
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
                

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    loss_name = "/Multi_InfoLoss.py"
    loss_log_dir = os.path.join(model_path, "loss")
    record_losses = LossLogger(log_dir=loss_log_dir)
    copyfile('./sample4geo' + loss_name, loss_log_dir + loss_name)
    print("\ncopy loss file from ./sample4geo" + loss_name)


    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=multi_info_loss,
                           optimizer=optimizer,
                           epoch=epoch,
                           record_losses=record_losses,
                           scheduler=scheduler,
                           scaler=scaler)

        record_losses.end_epoch()
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        for group in optimizer.param_groups:
            if any(p is model.logit_scale1 for p in group['params']):
                print(f"logit_scale1_lr: {group['lr']:.6f}")
            if any(p is model.logit_scale2 for p in group['params']):
                print(f"logit_scale2_lr: {group['lr']:.6f}")
        print("logit_scale1:{}, logit_scale2:{}".format(model.logit_scale1.item(), model.logit_scale2.item()))
        
        # evaluate
        #if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        if (epoch % config.eval_every_n_epoch == 0 and epoch > 40) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if config.save_weights and best_score > 0.955:
                    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                        torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(config.weights_save_path, epoch, r1_test))
                    else:
                        torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(config.weights_save_path, epoch, r1_test))
                    with open('{}/results.txt'.format(model_path), 'a') as f:
                        f.write('epoch={}, R@1={:.4f}\n'.format(epoch, r1_test*100))
                else:
                    with open('{}/results.txt'.format(model_path), 'a') as f:
                        f.write('epoch={}, R@1={:.4f}\n'.format(epoch, r1_test*100))

    # 保存最后一个epoch的权重
    if config.save_weights:
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(config.weights_save_path))
        else:
            torch.save(model.state_dict(), '{}/weights_end.pth'.format(config.weights_save_path))

    # 训练结束后保存最终结果
    summary = record_losses.finalize()
    print("\nTraining complete!")
    print(f"Epoch loss log saved to: {os.path.join(record_losses.log_dir, 'loss_log.txt')}")
    print("Training summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
