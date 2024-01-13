from cProfile import label
import os, shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import datetime
from model import TEC_LSTM, TEC_GRU, TEC_CNNGRU, Transformer, Multi_Transformer

models = {'LSTM':TEC_LSTM, 'GRU':TEC_GRU, 'CNNGRU':TEC_CNNGRU, 'Transformer':Transformer, "Multi_Transformer":Multi_Transformer}
# save train or validation loss
def log_loss(loss_val : float, path_to_save_loss : str, train : bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val)+"\n")
        f.close()

def EMA(values, alpha=0.1):
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha*item + (1-alpha)*ema_values[idx])
    return ema_values

def plot_tec(pred, target, show_type, dates):
    title = f'TEC - Predict v.s. Target {show_type}'
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({"figure.figsize":[14, 10]})
    plt.plot(pred, color = 'indigo', label='Prediction')
    plt.plot(target, '--', label="Target")
    plt.minorticks_on()
    plt.xticks([i for i in range(0, len(target), 8)], [dates[i] for i in range(0, len(dates), 8)])
    plt.xlabel('Time (Hour)')
    plt.ylabel('TEC')
    plt.legend()
    plt.title(title)
    plt.savefig(f'img/{title}.png')
    plt.close()

def plot_tec_pred(target, error, error_, omni, DOY, datetimes=[0,0,0,0], use_model=''):
    '''
    error_label = ['RMSE Global', 'RMSE Taiwan', 'RMSE North Pole', 'RMSE Sourth Pole', 'MAPE Global',\
     'MAPE Taiwan', 'MAPE North Pole', 'MAPE Sourth Pole', 'Max Error', 'Min Error', 'Pred TEC', 'True TEC']
     ['RMSE Global', 'RMSE Taiwan', 'MAPE Global', 'MAPE Taiwan', 'Max Error', 'Min Error']
    omni_label = ['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.']
    '''
    error_, omni = np.expand_dims([error_[0][0], error_[0][1], error_[0][4], error_[0][5], error_[0][8], error_[0][9]], axis=1), np.expand_dims(omni[0], axis=1)
    #print(error_, omni)
    date_format = f'{datetimes[0]}-{datetimes[1]:02d}-{datetimes[2]:02d} {datetimes[3]:02d}'
    ori_size = plt.rcParams["figure.figsize"]
    plt.figure(figsize=(ori_size[0]*3, ori_size[1]*3))
    plt.rcParams['font.size'] = '40'
    #plt.figure(figsize=(24, 12))
    #fig = plt.figure()
    #fig.set_figheight(plt.rcParams["figure.figsize"][1]+2)
    plt.style.use('seaborn-ticks')

    cmap = plt.cm.get_cmap('jet')
    ax1 = plt.subplot(221)
    ax1.margins(2, 2)      
    norm = mpl.colors.Normalize(vmin=0, vmax=350)    
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax1.imshow(target, cmap = cmap, interpolation='nearest', extent=[-180, 180, -87.5, 87.5], aspect=2)
    ax1.set_title(f'{use_model}')

    cmap = plt.cm.get_cmap('RdBu')
    ax2 = plt.subplot(222)
    ax2.margins(2, 2)       
    norm = mpl.colors.Normalize(vmin=-50, vmax=85) 
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))  
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax2.imshow(error, cmap = cmap, interpolation='nearest', extent=[-180, 180, -87.5, 87.5], aspect=2)
    ax2.set_title(f'{use_model}-GIM')

    ax3 = plt.subplot(223)
    ax3.axis('off')
    ax3.axis('tight')
    t1 = ax3.table(cellText=omni,
                    colWidths=[0.2]*6,
                    cellLoc='center',
                    rowColours =["palegreen"] * 6,
                    rowLabels=['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.', 'GMS size'],
                    bbox=[0.58, 0.01, 0.2, 1])
    t1.scale(1, 1.15)
    t1.set_fontsize(40)
    ax3.set_title(f'Space weather', x=0.58, fontsize=40)
    
    ax4 = plt.subplot(224)
    ax4.axis('off')
    t2 = ax4.table(cellText=error_,
                    colWidths=[0.2]*6,
                    cellLoc='center',
                    rowColours=["palegreen"] * 6,
                    rowLabels=['RMSE Global', 'RMSE Taiwan', 'MAPE Global', 'MAPE Taiwan', 'Max Error', 'Min Error'],
                    bbox=[0.59, 0.01, 0.2, 1])
    t2.scale(1, 1.15)
    t2.set_fontsize(40)
    ax4.set_title(f'Observation and Error', x=0.59, fontsize=40)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.suptitle(f'{use_model} Prediction: {datetimes[0]} Day {DOY:03d} UTC{datetimes[3]:02d}:00', y=0.98, fontsize=60)
    if use_model == 'obs': plt.savefig(f'img/observation/{DOY}->{date_format}.png') 
    else: plt.savefig(f'img/pred/{DOY}->{date_format}.png')
    plt.close('all')

def information_to_csv(errors, omnis, times):
    error_dict = {
        'DOY': times[:, 0],
        'Hour': times[:, 1],
        'Taiwan TEC':errors[:, -1],
        'Global TEC':errors[:, -3],
        'North Pole TEC':errors[:, -4],
        'Sourth Pole TEC':errors[:, -5],
        'RMSE Global':errors[:, 0],
        'RMSE Taiwan':errors[:, 1],
        'RMSE North Pole':errors[:, 2],
        'RMSE Sourth Pole':errors[:, 3],
        'Dst index':omnis[:, 0],
        'f10.7 index':omnis[:, 1],
        'Kp index':omnis[:, 2]
    }
    real_data = {
        'Global TEC':errors[:, -3],
        'Taiwan TEC':errors[:, -1],
        'North Pole TEC':errors[:, -4],
        'Sourth Pole TEC':errors[:, -5],
        'Dst index':omnis[:, 0],
        'f10.7 index':omnis[:, 1],
        'Kp index':omnis[:, 2]
    }
    plt.figure(figsize=(14, 9))
    plt.rcParams["figure.autolayout"] = True    
    error_obs = pd.DataFrame.from_dict(error_dict)
    error_obs.to_csv('observation_error_corr.csv', index=False)
    error_obs_corr = error_obs.corr()
    print(error_obs_corr)
    df = pd.DataFrame.from_dict(real_data)
    corr = df.corr()
    print(corr)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    cor_matrix = sn.heatmap(corr, 
        mask=mask,
        xticklabels=corr.columns,
        yticklabels=corr.columns, 
        square=True)#cmap='binary'
    cor_matrix.set_yticklabels(cor_matrix.get_yticklabels(), rotation = 0, fontsize = 20)
    cor_matrix.set_xticklabels(cor_matrix.get_xticklabels(), rotation = 75, fontsize = 20)
    plt.tight_layout() 
    cor_matrix.get_figure().savefig("img/line_chart/cor_matrix.png") 
               

def plot_trend(errors, omni_data, DOY_label):
    print(DOY_label)
    mpl.rcParams.update(mpl.rcParamsDefault)
    ori_size = plt.rcParams["figure.figsize"]
    print(ori_size)
    plt.figure(figsize=(ori_size[0]*2, ori_size[1]*3))
    error_label = ['RMSE Global', 'RMSE Taiwan', 'RMSE North Pole', 'RMSE Sourth Pole', 'MAPE Global',\
     'MAPE Taiwan', 'MAPE North Pole', 'MAPE Sourth Pole', 'Max Error', 'Min Error', 'Pred TEC', 'True TEC']
    omni_label = ['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.']
    fig, axs = plt.subplots(4)
    #plt.rcParams.update({'font.size': 25})
    #fig.suptitle(f'The line chart of trend') 
    axs[0].plot(errors[:, -1], label=error_label[-1][:4])
    axs[0].plot(errors[:, -2], label=error_label[-2][:4])
    axs[0].plot(errors[:, 1], label=error_label[1][:5]) 
    axs[0].set_title(f'Taiwan TEC') #TEC
    axs[0].legend(loc='upper right')

    axs[1].plot(errors[:, 0], label=error_label[0][5:])
    axs[1].plot(errors[:, 2], label=error_label[2][5:])
    axs[1].plot(errors[:, 3], label=error_label[3][5:])
    
    axs[1].set_title(f'{error_label[0][:4]}') #RMSE
    axs[1].legend(loc='upper right')

    #axs[2].plot(errors[:, 4], label=error_label[4][5:])
    #axs[2].plot(errors[:, 6], label=error_label[6][5:])
    #axs[2].plot(errors[:, 7], label=error_label[7][5:])
    #axs[2].set_title(f'{error_label[4][:4]}') #MAPE
    #axs[2].legend(loc='upper right')

    axs[2].plot(omni_data[:, 0])
    axs[2].set_title(f'{omni_label[0]}') #Dst

    #axs[3].plot(omni_data[:, 1])
    #axs[3].set_title(f'{omni_label[1]}') #f10.7
    axs[3].plot(omni_data[:, 2], label=omni_label[2][:2])
    axs[3].plot(omni_data[:, 3], label=omni_label[3][:2])
    axs[3].set_title(f'{omni_label[2]} & {omni_label[3]}') #Kp, ap
    axs[3].legend(loc='upper left')
    #axs[5].plot(omni_data[:, 4])
    #axs[5].set_title(f'{omni_label[4]}') #R
    plt.setp(axs, xticks=[i for i in range(0, 300, 25)], xticklabels=DOY_label)#
    fig.tight_layout()
    fig.savefig(f'img/line_chart/line_chart.png')
    plt.close('all')

def plot_tec_table(table, month, year):
    title = f'Relation between day and hour - {year}_MONTH_{month}'
    df_cfm = pd.DataFrame(table, index = [str(i) for i in range(len(table))], columns = [str(i) for i in range(1, len(table[-1])+1)])
    plt.figure(figsize=(24, 12))
    plt.rcParams.update({'font.size': 25})
    #sn.set(font_scale=2) # for label size
    cfm_plot = sn.heatmap(df_cfm, annot=False, annot_kws={"size": 25}, linewidths=.5, vmin=0, vmax=140)
    plt.xlabel('Days of Month')
    plt.ylabel('Hour')
    plt.title(title)
    cfm_plot.figure.savefig(f'img/{year}-month_{month}.png')
    plt.close('all')

def plot_rmse(values, show_type, dates):
    title = f'TEC - RMSE:abs(pred-target) {show_type}'
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({"figure.figsize":[14, 10]})
    plt.plot(values, color = 'indigo', label='RMSE')
    plt.minorticks_on()
    #plt.xticks([i for i in range(0, len(values), 50)], [dates[i] for i in range(0, len(dates), 50)])
    plt.xticks([i for i in range(len(values))], [i for i in range(0, 24)])
    plt.xlabel('Time (Hour)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(title)
    plt.savefig(f'img/{title}.png')
    plt.close()

def plot_loss(path_save_loss, train=True):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10,6.8))
    with open(path_save_loss+'train_loss.txt', 'r') as f:
        train_loss_list = [float(line) for line in f.readlines()]
    with open(path_save_loss+'val_loss.txt', 'r') as f:
        val_loss_list = [float(line) for line in f.readlines()]
    #if train:title = "Train"
    #else:title = "Validation"
    title = "Training v.s. Validation Loss"
    #train_EMA_loss = EMA(train_loss_list)
    #val_EMA_loss = EMA(val_loss_list)
    plt.plot(train_loss_list, color = 'indigo', label='train_Loss')
    #plt.plot(train_EMA_loss, '--', label="EMA train_loss")
    plt.plot(val_loss_list, color = 'red', label='valid_Loss')
    #plt.plot(val_EMA_loss, '--', label="EMA valid_loss")
    plt.minorticks_on()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title+' Loss')
    plt.savefig(path_save_loss+f'{title}.png')
    plt.close()

def clean_directory():
    if os.path.exists('save_loss'):
        shutil.rmtree('save_loss')
    if os.path.exists('save_model'): 
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions'): 
        shutil.rmtree('save_predictions')
    if os.path.exists('img/real'):
        shutil.rmtree('img/real')
    if os.path.exists('img/pred'): 
        shutil.rmtree('img/pred')
    if os.path.exists('img/GIM-Pred'): 
        shutil.rmtree('img/GIM-Pred')
    os.mkdir("save_loss")
    os.mkdir("save_model")
    os.mkdir("save_predictions")
    os.mkdir('img/real')
    os.mkdir('img/pred')
    os.mkdir('img/GIM-Pred')

if __name__ == "__main__":
    #plot_loss('save_loss/')
    from PIL import Image
    truth = os.listdir('img/observation/')
    pred = os.listdir('img/merge/')
    for fn in truth: 
        image1 = Image.open(f'img/observation/{fn}')
        image2 = Image.open(f'img/pred/{fn}')
        #resize, first image
        image1 = image1.resize((420, 300))
        image2 = image2.resize((500, 300))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB',(image1_size[0]+image2_size[0], image1_size[1]), (250,250,250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(image1_size[0],0))
        new_image.save(f"img/merge/{fn}","PNG")