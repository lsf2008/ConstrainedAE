'''
Description: 
Autor: Shifeng Li
Date: 2022-10-08 20:08:56
LastEditTime: 2022-10-15 07:49:49
'''
import torch

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ped2.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/8 20:08   shifengLi     1.0         None
'''

# import lib
import time
import main
import utils

from trainer.mult_ae_td_one_trainer import MulitAeTdOneTrainer
# from pytorch_lightning import  M
import pytorch_lightning as pl
pl.seed_everything(999999, workers=True)
def ped2Train(svPth='data/result/ped2_train.pt'):
    pl.seed_everything(999999, workers=True)
    args, train_cfg, dt_cfg = utils.initial_params(train_cfg='config/ped2_train_cfg.yaml', dt_cfg='config/dtped2_cfg.yml')
    # model=AeMultiOut(input_shape=train_cfg['input_shape'], code_length=train_cfg['code_length'])
    # train_cfg['model'] = model
    res = main.main_train(args, train_cfg, dt_cfg,
                        trainer = MulitAeTdOneTrainer,
                        sv_auc_pth='data/ped2_val_auc.pt')

    torch.save(res, svPth)

def ped2Tst():
    # AeTdOneTrainerModel.load
    model = MulitAeTdOneTrainer.load_from_checkpoint(
        checkpoint_path="data/model/ped2_aetdsvdd.ckpt",
        hparams_file="data/model/ped2_aetdsvdd_hparams.yaml",
        map_location=None,
    )
    trainer = MulitAeTdOneTrainer()
    trainer.test(model)


if __name__=='__main__':
    st_time=time.time()
    ped2Train()

    # res = torch.load('data/result/ped2_params.pt')
    end_time = time.time()
    print(f'run time : {(end_time-st_time)/60}m')

