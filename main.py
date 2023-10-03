import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # main root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from tqdm import tqdm
from copy import deepcopy

from utils.general import set_seed
from utils.dataloader import CustomDataLoader
from models.tsmixer import TSMixerRevIN


def main(args):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    set_seed(args.seed)

    # load datasets
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )

    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()

    # load model
    model = TSMixerRevIN(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
#         norm_type=args.norm_type,
#         activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
    ).to(device)

    # set criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = torch.tensor(float('inf'))

    # create checkpoint directory
    save_directory = os.path.join(args.checkpoint_dir, args.name)

    if os.path.exists(save_directory):
        import glob
        import re

        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        save_directory = f"{path}{n}"  # update path

    os.makedirs(save_directory)

    # start training
    for epoch in range(args.train_epochs):
    
        model.train()
        
        train_mloss = torch.zeros(1)
        
        print(('\n' + '%-10s' * 2) % ('Epoch', 'Train loss'))
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        for i, (batch_x, batch_y) in pbar:
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            optimizer.zero_grad()
            
            loss = criterion(outputs, batch_y)
                
            loss.backward()
            
            optimizer.step()
            
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)

            pbar.set_description(('%-10s' * 1 + '%-10.4g' * 1) %
                                 (f'{epoch+1}/{args.train_epochs}', train_mloss))

            # end batch -------------------------------------------------------------

        model.eval()
        
        val_mloss = torch.zeros(1)
        
        print(('%-10s' * 2) % ('', 'Val loss'))
        pbar = tqdm(enumerate(val_data), total=len(val_data))
        
        with torch.no_grad():
            
            for i, (batch_x, batch_y) in pbar:

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)

                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)

                pbar.set_description(('%-10s' * 1 + '%-10.4g' * 1) %
                                     (f'', val_mloss))


            if val_mloss < best_loss:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())

                torch.save(best_model, os.path.join(save_directory, "best.pt"))
                
                patience = 0
            else:
                patience += 1
                
                if patience == args.patience:
                    break

        # end epoch -------------------------------------------------------------


    # load best model
    model.load_state_dict(best_model)

    # start testing
    model.eval()
    
    test_mloss = torch.zeros(1)

    print(('\n' + '%-10s' * 1) % ('Test loss'))
    pbar = tqdm(enumerate(val_data), total=len(val_data))

    with torch.no_grad():

        for i, (batch_x, batch_y) in enumerate(tqdm(test_data)):

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)

            pbar.set_description(('%-10.4g' * 1) %
                                 (test_mloss))


def parse_args():

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    parser.add_argument('--data', 
                        type=str, 
                        default=ROOT / 'data/weather.csv', 
                        help='dataset.csv path')
    parser.add_argument(
        '--feature_type',
        type=str,
        default='M',
        choices=['S', 'M', 'MS'],
        help=(
            'forecasting task, options:[M, S, MS]; M:multivariate predict'
            ' multivariate, S:univariate predict univariate, MS:multivariate'
            ' predict univariate'
        ),
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=ROOT / 'checkpoints',
        help='location of model checkpoints',
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='save best model to checkpoints/name',
    )

    # forecasting task
    parser.add_argument(
        '--seq_len', type=int, default=336, help='input sequence length'
    )
    parser.add_argument(
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )

    # model hyperparameter
    parser.add_argument(
        '--n_block',
        type=int,
        default=2,
        help='number of block for deep architecture',
    )
    parser.add_argument(
        '--ff_dim',
        type=int,
        default=2048,
        help='fully-connected feature dimension',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.05, help='dropout rate'
    )
    parser.add_argument(
        '--norm_type',
        type=str,
        default='B',
        choices=['L', 'B'],
        help='LayerNorm or BatchNorm',
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu'],
        help='Activation function',
    )

    # optimization
    parser.add_argument(
        '--train_epochs', type=int, default=100, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size of input data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='optimizer learning rate',
    )
    parser.add_argument(
        '--patience', type=int, default=5, help='number of epochs to early stop'
    )

    # save results
    parser.add_argument(
        '--result_path', default='result.csv', help='path to save result'
    )

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)