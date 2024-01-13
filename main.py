import argparse
from train_model import *
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from inference import *
import faulthandler
faulthandler.enable()

def main(
    epoch = 10,
    batch_size = 32,
    train_path = 'txt/train/',
    path_save_model = 'save_model/',
    path_save_loss = 'save_loss/',
    device='cpu',
    use_model='LSTM', 
    to_sequence=False
):
    window_sz = 1
    in_dim, out_dim = 72*window_sz*1+6, 72 #
    '''
    _input, target, date = get_data(train_path)
    x_train, x_test, y_train, y_test, date_train, date_test =\
    train_test_split(_input, target, date, test_size=0.2, random_state=2022)

    x_train, x_val, y_train, y_val, date_train, date_val =\
    train_test_split(x_train, y_train, date_train, test_size=0.2, random_state=2022)

    print(x_train.shape, x_val.shape, x_test.shape)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader([x_train, y_train, date_train], [x_val, y_val, date_val], [x_test, y_test, date_test], batch_size)
    '''
    train_dataset = TecDataset(train_path, window_size=window_sz, to_sequence=to_sequence)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle = True)
    valid_dataset = TecDataset('txt/2020/', data_type='dir', mode='val', window_size=window_sz, to_sequence=to_sequence)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
    #test_dataset = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    test_dataset = TecDataset('txt/2021/', data_type='dir', mode='test', window_size=window_sz, to_sequence=to_sequence)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    print('done')
    clean_directory()
    #if "Transformer" in use_model:
    #    best_model = train_transformer(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, epoch, path_save_model, path_save_loss, device, use_model)
    #else:
    #    best_model = train(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, epoch, path_save_model, path_save_loss, device, use_model)  
    
    best_model = train(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, epoch, path_save_model, path_save_loss, device, use_model)  
    #best_model = train_with_discriminator(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, epoch, path_save_model, path_save_loss, device, use_model)
    inference(path_save_model, in_dim, out_dim, best_model, test_dataloader, device, use_model)
    #inference('save_model/', in_dim, out_dim, 'best_train_31_Transformer.pth', tmpdataloader, device, 'Transformer')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_path', type=str, default='txt/train/')
    parser.add_argument('--path_save_model', type=str, default='save_model/')
    parser.add_argument('--path_save_loss', type=str, default='save_loss/')
    parser.add_argument('--use_model', type=str, default='LSTM')
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    main(epoch=args.epoch, batch_size=args.batch_size, train_path=args.train_path, path_save_model=args.path_save_model,\
     path_save_loss=args.path_save_loss, device=device, use_model=args.use_model, to_sequence=True)
