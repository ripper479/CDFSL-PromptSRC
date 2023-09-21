import numpy as np
import torch
import torch.optim
import os

from data.datamgr import SetDataManager
from methods.prompt_src import PromptSRC
from options import parse_args
import time, json

def get_classnames(data_file):
    with open(data_file, 'r') as f:
      meta = json.load(f) 
      return meta['label_names']    


def train(base_loader, val_loader, trainer, start_epoch, stop_epoch, params):
    total_it = 0
    for epoch in range(start_epoch, stop_epoch):
        since = time.time()
        trainer.model.train()
        total_it = trainer.train_loop(epoch, base_loader, total_it)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        trainer.model.eval()
        with torch.no_grad():
            acc = trainer.test_loop(val_loader)
        
        print("Epoch {:f} Accuracy {:f}".format(epoch,acc))

    trainer.save_model(params.save_dir)

        

# --- main function ---
if __name__=='__main__':
    # set numpy random seed
    np.random.seed(10)

    # parser argument
    params = parse_args()
    print('--- Training ---\n')
    print(params)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare dataloader ---')
    print('\ttrain with seen domain {}'.format(params.dataset))
    print('\tval with seen domain {}'.format(params.testset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    classnames = get_classnames(base_file)
    val_file = os.path.join(params.data_dir, params.testset, 'val.json')


    image_size = 224
    n_query = max(1, int(16*params.test_n_way/params.train_n_way))
    base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # trainer
    trainer = PromptSRC(n_way=params.train_n_way, n_support=params.n_shot,classnames=classnames)
    trainer.build_model()

    # training
    print('\n--- start the training ---')
    train(base_loader, val_loader, trainer, start_epoch, stop_epoch, params)