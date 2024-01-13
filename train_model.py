from pickletools import optimize
from model import TEC_LSTM, TEC_GRU, TEC_CNNGRU
from torch.utils.data import DataLoader
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from dataloader import TecDataset
import logging, random
from inference import inference
from helper import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def flip_from_probability(p):
    return True if random.random() < p else False

def MAELoss(output, target):
    mask = target != 0
    return torch.mean(torch.abs((target - output)))

def evaluation(dataloader, model, device, use_model='Transformer_CNN'):
    model.eval()    
    val_loss = 0
    mse = torch.nn.MSELoss()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        b_information = batch[3].to(device)
        output = model(torch.cat((b_input, b_information), 2))
        #output = model(b_input)
        loss = torch.sqrt(mse(output, b_target))
        #loss = mse(output, b_target)
        #loss = torch.sqrt(mse(output[3], b_target[:, 3, :, :]))
        val_loss += loss.detach().item()
    return val_loss / len(dataloader)

def train_transformer(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, path_save_loss, device, use_model='Transformer_CNN'):
    same_seeds(0)
    clean_directory()
    device = torch.device(device)    
    model = models[use_model](in_dim, out_dim, batch_size, device).float().to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        #training
        model.train()
        for step, batch in enumerate(dataloader):            
            #optimizer.zero_grad()
            for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            b_information = batch[3].to(device)
            if use_model == 'Transformer_CNN':
                output = model(b_input)
            else:
                #output = model(b_input)
                output = model(torch.cat((b_input, b_information), 2))
            # RMSE loss function
            loss = torch.sqrt(mse(output, b_target))

            # MAPE loss function
            #loss = MAELoss(output, b_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_{use_model}.pth')
            #torch.save(optimizer.state_dict(), path_save_model + f'optimizer_{epoch}_{use_model}.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_{use_model}.pth"

        train_loss /= len(dataloader)
        val_loss = evaluation(valid_dataloader, model, device)
        scheduler.step(val_loss)
        log_loss(train_loss, path_save_loss, train=True)
        log_loss(val_loss, path_save_loss, train=False)
        #val_loss = evaluation(val_dataloader, model, device), Validation loss: {val_loss:5.5f}
        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.3f}, Validation loss: {val_loss:5.3f}")

    plot_loss(path_save_loss)
    return best_model

def train(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, path_save_loss, device, use_model='LSTM'):#, val_dataloader
    same_seeds(0)
    clean_directory()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, batch_size, device).float().to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        #training
        model.train()
        for step, batch in enumerate(dataloader):            
            #optimizer.zero_grad()
            for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            b_information = batch[3].to(device)
            if use_model == 'CNNGRU':
                output = model(b_input.unsqueeze(1))
            elif use_model == 'Multi_Transformer':
                output = model(b_input)
                #output = model(torch.cat((b_input, b_information), 2))
            else: #output = model(torch.cat((b_input, b_information), 2))
                #output = model(b_input)
                output = model(torch.cat((b_input, b_information), 2))
            # MAPE loss function
            #loss = MAELoss(tec_output, b_target)
            # RMSE loss function
             
            loss = torch.sqrt(mse(output, b_target))        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_{use_model}.pth')
            #torch.save(optimizer.state_dict(), path_save_model + f'optimizer_{epoch}_{use_model}.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_{use_model}.pth"

        train_loss /= len(dataloader)
        val_loss = evaluation(valid_dataloader, model, device)
        scheduler.step(val_loss)
        log_loss(train_loss, path_save_loss, train=True)
        log_loss(val_loss, path_save_loss, train=False)
        #val_loss = evaluation(val_dataloader, model, device), Validation loss: {val_loss:5.5f}
        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.3f}, Validation loss: {val_loss:5.3f}")

    plot_loss(path_save_loss)
    return best_model

def train_with_discriminator(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, path_save_loss, device, use_model='LSTM'):#, val_dataloader
    same_seeds(0)
    clean_directory()
    device = torch.device(device)    
    model = models[use_model](in_dim, out_dim, batch_size, device).float().to(device)
    print(model)
    discriminator = Discriminator().float().to(device)
    print(discriminator)
    D_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
    criterion = torch.nn.MSELoss()#nn.HingeEmbeddingLoss()
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss, total_G_loss, total_D_loss = 0, 0, 0
        #training
        model.train()
        discriminator.train()
        for step, batch in enumerate(dataloader):            
            #for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            b_information = batch[3].to(device)
            omni = batch[4].float().to(device)
            #tec_output = model(torch.cat((b_input, b_information), 2), omni)
            tec_output = model(torch.cat((b_input, b_information), 2))
            
            # discriminator output
            real_score = discriminator(b_target.unsqueeze(1))
            # compute loss
            loss_D = torch.sqrt(criterion(real_score, omni))
            total_D_loss += loss_D

            # update model
            discriminator.zero_grad()
            loss_D.backward()
            D_optimizer.step()
    
            tec_output = model(torch.cat((b_input, b_information), 2))
            # dis
            f_logit = discriminator(tec_output.unsqueeze(1))
            
            # compute loss
            loss_G = torch.sqrt(criterion(f_logit, omni))
            tec_loss = torch.sqrt(mse(tec_output, b_target))
            loss = tec_loss + loss_G

            # update model    
            model.zero_grad()    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += tec_loss.detach().item()
            total_G_loss += loss_G.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_{use_model}.pth')
            #torch.save(optimizer.state_dict(), path_save_model + f'optimizer_{epoch}_{use_model}.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_{use_model}.pth"

        train_loss /= len(dataloader)
        total_G_loss /= len(dataloader)
        total_D_loss /= len(dataloader)
        val_loss = evaluation_for_dis(valid_dataloader, model, device)
        scheduler.step(val_loss)
        log_loss(train_loss, path_save_loss, train=True)
        log_loss(val_loss, path_save_loss, train=False)
        #val_loss = evaluation(val_dataloader, model, device), Validation loss: {val_loss:5.5f}
        logger.info(f"Epoch: {epoch:03d}, Training loss: {train_loss:02.3f}, Validation loss: {val_loss:02.3f}, G loss: {total_G_loss:02.3f}, D loss: {total_D_loss:02.3f}")

    plot_loss(path_save_loss)
    return best_model

if __name__ == '__main__':
    plot_loss('save_loss/')


