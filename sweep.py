import wandb


#For hyper parameter tuning

path = '/data/code/survival_prediction/configs/sweeps/vit_pretrain_sweep.yaml'  #config file 

sweep_id = wandb.sweep(sweep=path, project="sweeps") #save in wandb
