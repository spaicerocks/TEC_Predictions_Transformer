from model import TEC_LSTM, TEC_GRU
from torch.utils.data import DataLoader
import torch, random
import torch.nn as nn
from dataloader import TecDataset
import logging
import pandas as pd
from helper import *
import matplotlib.pyplot as plt
from GIM_TXT_to_csv import plot, generate_gif, plot_init
import numpy as np
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def mape(y_pred, y_target):
    mask=y_target!=0
    return np.fabs((y_target[mask]-y_pred[mask])/y_target[mask])

def read_one_day_omni(all_omni, DOY, hour):   
    omni_data = all_omni.iloc[(DOY-1)*24+hour].to_list() 
    #print(omni_data)
    #Kp index,R (Sunspot No.),"Dst-index, nT","ap_index, nT",f10.7_index
    #['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.']
    data = list(omni_data[3:])
    return [data[2], data[4], data[0], data[3], data[1], np.nan]

def get_error(output, target):
    gim_flatten = torch.flatten(target[0]).cpu().detach().numpy()
    pred_flatten = torch.flatten(output[0]).cpu().detach().numpy()
    gim_mean = np.mean(gim_flatten)

    mse_loss = torch.nn.MSELoss()
    loss = torch.nn.MSELoss(reduction='none')
    taiwan_point = 25*72 + 61
    taiwan_tec = gim_flatten[taiwan_point]
    taiwan_pred = pred_flatten[taiwan_point]

    rmse_map = torch.flatten(torch.sqrt(loss(output[0], target[0]))).cpu().detach().numpy()

    polar_out_n, polar_out_s = output[0][:11], output[0][60:]
    polar_tar_n, polar_tar_s = target[0][:11], target[0][60:]
    polar_tar_n_mean, polar_tar_s_mean = np.mean(polar_tar_n.cpu().detach().numpy()), np.mean(polar_tar_s.cpu().detach().numpy())
    polar_rmse_n = torch.sqrt(mse_loss(polar_out_n, polar_tar_n)).cpu().detach().numpy()
    polar_rmse_s = torch.sqrt(mse_loss(polar_out_s, polar_tar_s)).cpu().detach().numpy()

    polar_mape_n = mape(polar_out_n.cpu().detach().numpy(), polar_tar_n.cpu().detach().numpy()).mean()
    polar_mape_s = mape(polar_out_s.cpu().detach().numpy(), polar_tar_s.cpu().detach().numpy()).mean()
    #print(polar_rmse_n, polar_rmse_s)
    #print(polar_mape_n, polar_mape_s)
    #input()

    error_map = torch.flatten(torch.sub(output[0], target[0])).cpu().detach().numpy()
    rmse = rmse_map[taiwan_point]
    mape_loss = mape(output.cpu().detach().numpy(), target.cpu().detach().numpy())
    global_rmse, global_mape = torch.sqrt(mse_loss(output, target)), mape_loss.mean()    
    local_rmse, local_mape = float(rmse), mape_loss[taiwan_point]
    max_error, min_error = max(error_map), min(error_map)
    return [float("%2.3f"%global_rmse.cpu().detach().numpy()), float("%2.3f"%local_rmse), float("%2.3f"%polar_rmse_n), float("%2.3f"%polar_rmse_s),\
     float("%2.3f"%global_mape), float("%2.3f"%local_mape), float("%2.3f"%polar_mape_n), float("%2.3f"%polar_mape_s), \
     float("%2.3f"%max_error), float("%2.3f"%min_error), float("%2.3f"%polar_tar_n_mean), float("%2.3f"%polar_tar_s_mean),\
     float("%2.3f"%gim_mean), float("%2.3f"%taiwan_pred), float("%2.3f"%taiwan_tec)]

def inference_four_hour(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    tec_tar, tec_pred, dates, errors, omnis = [], [], [], [], []
    taiwan_point, idx = 25*72 + 61, 0
    DOY_label, times = [], []
    omni_path = 'omni/2020hourly.csv'
    all_omni = pd.read_csv(omni_path)
    plot_init()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, 1, device).float().to(device)
    model.load_state_dict(torch.load(path_save_model+best_model))
    criterion = torch.nn.MSELoss()
    #x, y = random.randint(0, 73), random.randint(0, 71)
    data_list = list(dataloader)
    model.eval()
    while True:
        if idx >= len(data_list):break
        batch = data_list[idx]
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        b_information = batch[3].to(device)
        b_time = tuple(b.numpy()[0] for b in batch[2])  
        idx_4_h = int(idx)
        for _ in range(4):
            if idx_4_h >= len(data_list):break
            batch = data_list[idx_4_h]
            t_target, information = tuple(b.to(device) for b in [batch[1], batch[3]])
            output = model(torch.cat((b_input, information), 2))
            output = output.type(torch.LongTensor).type(torch.FloatTensor).to(device)
            t_time = tuple(b.numpy()[0] for b in batch[2]) 
            y, m, d, h = t_time[:]
            dates.append(f'{m}/{d}\n{h}:00') 
            mean, std = torch.mean(output), torch.std(output)
            prediction = (output-mean) / std  # 儲存model prediction, 並正規化  
            b_input = torch.cat((b_input[: ,: ,72:], prediction.detach()) ,2)

            pred = torch.tensor(output, dtype=int).clone().cpu().detach().numpy()
            #print(torch.argmax(torch.abs(torch.sub(b_target[0], output[0])), 1).cpu().detach().numpy())
            max_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(t_target[0], output[0]))), 60)[1].cpu().detach().numpy()
            min_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(t_target[0], output[0]))), 60, largest=False)[1].cpu().detach().numpy()
            max_points, min_points = [[idx%72, idx//72] for idx in max_idxes], [[idx%72, idx//72] for idx in min_idxes]
            points = [max_points, min_points]            
            
            tec_pred.append(torch.flatten(output.cpu()).detach().numpy()[taiwan_point])
            tec_tar.append(torch.flatten(t_target.cpu()).numpy()[taiwan_point])
            
            idx_4_h += 1
        error_ = get_error(output, t_target)
        
        target, model_pred, error = t_target.cpu().numpy()[0], pred[0], torch.sub(output[0], t_target[0]).cpu().detach().numpy()
        m, d, h = t_time[1:]
        DOY, hour = int(datetime.datetime.strptime(f'{y}-{m}-{d}', '%Y-%m-%d').strftime('%j')), int(h)
        print(DOY, hour)
        DOY_label.append(DOY)
        times.append((DOY, hour))
        omni = read_one_day_omni(all_omni, DOY, hour)
        errors.append(error_)
        omnis.append(omni)
        plot_tec_pred(model_pred, error, [error_], [omni], DOY, datetimes=t_time, use_model=use_model)   
        idx+=1
    #show_big_error(error_, DOY_label, 50)
    DOY_label = sorted(list(set(DOY_label)))
    print(DOY_label)
    print(np.array(errors))
    print(np.array(omnis))
    plot_trend(np.array(errors), np.array(omnis), DOY_label)
    information_to_csv(np.array(errors), np.array(omnis), np.array(times))
    plt.close('all')   
    #plot_tec(tec_pred, tec_tar, 'Taiwan', dates)            

def make_gif(path_list, timestamp, use_model):
    generate_gif(path_list[0], timestamp, use_model='GIM-Pred')
    generate_gif(path_list[1], timestamp, use_model)
    generate_gif(path_list[2], timestamp)

def inference(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    # target map, predict map, rmse, target date, another date format
    tec_tar, tec_pred, tec_rmse, dates, d_f = [], [], [], [], []
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    map_points = [8*72 + 23, 25*72 + 60, 35*72 + 18, 28*72 + 20, 66*72 + 69]
    monthly_tec = [[]for i in range(12)]
    pred_tec = [[]for i in range(12)]
    rmse_global = [[]for i in range(12)]
    monthly_tec_loacl = [[[]for i in range(12)]for _ in range(len(map_points))]
    pred_tec_loacl = [[[]for i in range(12)]for _ in range(len(map_points))]
    rmse_local = [[[]for i in range(12)]for _ in range(len(map_points))]
    omni_path = 'omni/2020hourly.csv'
    all_omni = pd.read_csv(omni_path)
    tec_dict, error_log = {}, {}
    errors, omnis, times = [], [], []
    rmse_list, error_list, error_list_local = [], [], [[]for i in range(len(map_points))]
    # each hour tec
    
    taiwan_point = 25*72 + 61
    past_month = 0 #record month
    plot_init()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, 1, device).float().to(device)
    model.load_state_dict(torch.load(path_save_model+best_model))
    
    #x, y = random.randint(0, 73), random.randint(0, 71)
    model.eval()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        
        b_information = batch[3].to(device)
        b_time = tuple(b.numpy() for b in batch[2])
        if len(b_time)>=4:y, m, d, h = np.array(b_time)[:, 0]
        else:y, m, d, h = b_time[0][:]
        
        if use_model == 'CNNGRU':
            output = model(b_input.unsqueeze(1))
        else:
            output = model(torch.cat((b_input, b_information), 2))
            #output = model(b_input)
        pred = (torch.tensor(output)).cpu().detach().numpy()
        with open(f'model_output/2021/{y:02d}-{m:02d}-{d:02d} {h:02d}:00.npy', 'wb') as f:
            np.save(f, pred)
        b_target = (torch.tensor(b_target))
        
        dates.append(f'{y:02d}-{m:02d}-{d:02d} {h:02d}:00')
        map_flatten = torch.flatten(b_target.cpu()).detach().numpy()
        map_flatten_pred = torch.flatten(output.cpu()).detach().numpy()
        for idx in range(len(map_points)):
            gim, pred_gim = map_flatten[map_points[idx]], map_flatten_pred[map_points[idx]]
            monthly_tec_loacl[idx][m-1].append(gim)
            pred_tec_loacl[idx][m-1].append(pred_gim)
            error_list_local[idx].append(gim-pred_gim)
            rmse_local[idx][m-1].append(np.sqrt((gim-pred_gim)**2))

        tec_pred.append(map_flatten[taiwan_point])
        tec_tar.append(torch.flatten(b_target.cpu()).numpy()[taiwan_point])
        monthly_tec[m-1].append(np.mean(b_target.mean().cpu().detach().numpy()))
        pred_tec[m-1].append(np.mean(output.mean().cpu().detach().numpy()))
        error_map = torch.sub(b_target, output).cpu().detach().numpy()
        rmse_error = np.sqrt((error_map**2).mean())
        rmse_global[m-1].append(rmse_error)
        #plot(error_map, points=None, rmse=rmse_error, datetime=[y, m, d, h], type_='Difference', use_model='')
        rmse_list.append(rmse_error)
        error_list.append(error_map.mean())
        print(y, m, d, h,rmse_error)
        '''
        error_ = get_error(output, b_target)
        target, model_pred, error = b_target.cpu().numpy()[0], pred[0], torch.sub(output[0], b_target[0]).cpu().detach().numpy()
        DOY, hour = int(datetime.datetime.strptime(f'{y}-{m}-{d}', '%Y-%m-%d').strftime('%j'))+1, int(h)
        omni = read_one_day_omni(all_omni, DOY, hour)
        #plot_tec_pred(target, error, [error_], [omni], datetimes=b_time, use_model=use_model) 
        error_ = get_error(output, b_target) 
        times.append((DOY, hour))
        errors.append(error_)  
        omnis.append(omni) 
    information_to_csv(np.array(errors), np.array(omnis), np.array(times))
    '''
    #print(monthly_tec)
    #print(pred_tec)
    plot_rmse(rmse_global, rmse_local, map_points, month, y, use_model)
    plot_average_bar(monthly_tec, pred_tec, month, y, use_model)
    plot_average_local_bar(monthly_tec_loacl, pred_tec_loacl, month, map_points, y, use_model)
    plot_error_dist(error_list, y, use_model)
    plot_error_dist_local(error_list_local, y, use_model)
    #input()
    tec_dict['dates'], tec_dict['rmse'] = dates, rmse_list
    pd.DataFrame.from_dict(tec_dict).to_csv('rmse_result.csv', index=False)
    plt.close('all')           

def plot_rmse(rmse_global, rmse_local, map_points, month, year, use_model):
    rmse_global = [np.mean(rmse_global[i]) for i in range(len(month))]
    rmse_local = [[np.mean(rmse_local[j][i]) for i in range(len(month))]for j in range(len(map_points))]
    local_rmse_data = {}
    labels = ['Canada', 'Taiwan', 'Ecuador', 'Cook Islands', 'Antarctica']
    print(rmse_global)
    for i in range(len(map_points)):
        print(rmse_local[i])
    input()
    colors = ['blue', 'green', 'red', 'purple', 'pink']    
    x = np.arange(len(month))
    width = 0.1    
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 18})   
    plt.bar(x, rmse_global, width, color='black', label='Global')
    for i in range(len(colors)):             
        plt.bar(x + width*(i+1), rmse_local[i], width, color=colors[i], label=labels[i])   
    plt.xticks(x + width*2, month)
    plt.ylabel('Average RMSE (10TEC)')
    plt.title(f'Average RMSE for each month-{year}')
    plt.legend(loc='upper left')
    plt.savefig(f'img/average_rmse_{year}.png')
    plt.close('all')

def plot_average_bar(monthly_tec, pred_tec, month, year, use_model):
    monthly_tec = [np.mean(monthly_tec[i]) for i in range(len(month))]
    pred_tec = [np.mean(pred_tec[i]) for i in range(len(month))]
    x = np.arange(len(month))
    width = 0.3
    plt.figure(figsize=(15, 8))
    plt.rcParams.update({'font.size': 18})
    plt.bar(x, monthly_tec, width, color='green', label='CODE')
    plt.bar(x + width, pred_tec, width, color='blue', label=use_model)
    plt.xticks(x + width / 2, month)
    plt.ylabel('Average TEC(10TEC)')
    plt.title(f'Average TEC for each month ({year})')
    plt.legend(loc='upper left')
    plt.savefig(f'img/average_bar_{year}.png')
    plt.close('all')

def plot_average_local_bar(monthly_tec_loacl, pred_tec_loacl, month, map_points, year, use_model):
    monthly_tec_loacl = [[np.mean(monthly_tec_loacl[j][i]) for i in range(len(month))]for j in range(len(map_points))]
    pred_tec_loacl = [[np.mean(pred_tec_loacl[j][i]) for i in range(len(month))]for j in range(len(map_points))]
    colors = ['blue', 'green', 'red', 'purple', 'pink']
    labels = ['Canada', 'Taiwan', 'Ecuador', 'Cook Islands', 'Antarctica']
    x = np.arange(len(month))
    width = 0.3
    
    for i in range(len(colors)):
        plt.figure(figsize=(15, 8))
        plt.rcParams.update({'font.size': 18})
        plt.bar(x, monthly_tec_loacl[i], width, color=colors[0], label='CODE')
        plt.bar(x + width, pred_tec_loacl[i], width, color=colors[1], label=use_model)
        plt.xticks(x + width/2, month)
        plt.ylabel('Average TEC(10TEC)')
        plt.title(f'Average TEC for each month ({labels[i]}_({year}))')
        plt.legend(loc='upper left')
        plt.savefig(f'img/average_bar_local_{labels[i]}_{year}.png')
        plt.close('all')

def plot_error_dist_local(errors, year, use_model):
    labels = ['Canada', 'Taiwan', 'Ecuador', 'Cook Islands', 'Antarctica']
    colors = ['blue', 'green', 'red', 'purple', 'pink']
    for i in range(len(errors)):        
        n, bins, patches=plt.hist(errors[i], color=colors[i], bins=150, edgecolor='black', linewidth=0.8, label=labels[i])
        plt.xlabel("Forecast Error(10TEC)")
        plt.ylabel("Error Distribution")
        plt.title(f"Forecast error distribution ({labels[i]})_({year})")
        plt.legend(loc='upper right')
        plt.savefig(f'img/error_dist_{labels[i]}_{year}.png')
        plt.close('all')

def plot_error_dist(errors, year, use_model):
    n, bins, patches=plt.hist(errors, bins=150, edgecolor='black', linewidth=0.8, label=use_model)
    plt.xlabel("Forecast Error(10TEC)")
    plt.ylabel("Error Distribution")
    plt.title(f"Forecast error distribution_{year}")
    plt.legend(loc='upper right')
    plt.savefig(f'img/error_dist_{year}.png')
    plt.close('all')

def show_big_error(rmse, dates, value):
    for e, d in zip(rmse, dates):
        if e >= value:print(d, e)


if __name__ == '__main__':
    #clean_directory()
    window_sz = 1
    in_dim, out_dim = 72*window_sz+6, 72
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    #tmpdata = TecDataset('txt/monthly_testdata/', data_type='dir', mode='test')
    
    tmpdata = TecDataset('txt/2020/', data_type='dir', mode='test', window_size=window_sz, to_sequence=False)
    tmpdataloader = DataLoader(tmpdata, batch_size = 1, shuffle = False)
    inference('save_model/', in_dim, out_dim, 'best_train_43_Transformer.pth', tmpdataloader, device, 'Transformer')
    #inference_four_hour('save_model/', in_dim, out_dim, 'best_train_44_Transformer.pth', tmpdataloader, device, 'Transformer')
    