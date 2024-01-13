import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import geopandas
from matplotlib import cm
import pandas as pd
import datetime
import imageio, re
import seaborn as sn
from helper import plot_tec_pred

def plot_init():
    plt.rcParams.update({"figure.figsize":[10, 6]})
    plt.rcParams.update({"figure.autolayout":False}) 
    plt.rcParams.update({'font.size': 15})   
    #plt.xlim(-180, 180)
    #plt.ylim(-87.5, 87.5)

def plot(np_data, points=None, rmse=0, datetime=[0,0,0,0], type_='Code GIM', use_model=''):
    plt.figure(figsize=(16, 8))
    plt.rcParams.update({'font.size': 22})
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(figsize=(16, 8), edgecolor='black')
    if type_=='Difference':
        #cmap = plt.cm.get_cmap('jet') 
        #cmap = plt.cm.get_cmap('binary')   
        #cmap = plt.cm.get_cmap('RdBu') 
        cmap = plt.cm.get_cmap('Oranges')   
    else:
        cmap = plt.cm.get_cmap('jet') 

    date_format = f'{datetime[0]}-{datetime[1]:02d}-{datetime[2]:02d} {datetime[3]:02d}'
    if type_=='Difference':plt.title(f"GLOBAL IONOSPHERE MAPS\n{date_format}:00 UT\n{use_model} {type_} rmse = {rmse:02.3f}") 
    else:plt.title(f"GLOBAL IONOSPHERE MAPS\n{date_format}:00 UT\n{use_model} {type_}")    
     
    if points:
        for i in range(len(points[0])):plt.scatter(points[0][i][0], points[0][i][1], marker='o', color='m', label='diff_max'if i==0 else'')
        for i in range(len(points[1])):plt.scatter(points[1][i][0], points[1][i][1], marker='o', color='w', label='diff_min'if i==0 else'')        
        plt.legend(loc='upper right')
    
    
    im = plt.imshow(np_data, extent=[-180, 180, -87.5, 87.5], cmap = cmap)
    #if points:im = plt.imshow(np_data, cmap = cmap)
    #else:im = plt.imshow(np_data, extent=[-180, 180, -87.5, 87.5], cmap = cmap)
    #norm = mpl.colors.BoundaryNorm(list(range(0,400,50)), cmap.N)
    norm = mpl.colors.Normalize(vmin=0, vmax=350)   
    if points:plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    else:plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.8)
    #plt.colorbar(im, shrink=0.65)
    plt.xlabel('GEOGRAPHIC LONGITUDE')
    plt.ylabel('GEOGRAPHIC LATITUDE')
    if type_ == 'Target':plt.savefig(f'img/real/{date_format}.png')
    elif type_=='Input':plt.savefig(f'img/input/{date_format}.png')
    elif type_ == 'Difference':plt.savefig(f'img/difference/{date_format}.png')
    else:plt.savefig(f'img/pred/{date_format}.png')
    
    plt.close('all')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

def generate_gif(img_dir, datetime, use_model=''):
    filenames = os.listdir(img_dir)
    filenames.sort(key = natural_keys)
    with imageio.get_writer(f'gif/{datetime[0]}-{datetime[1]:02d}-{datetime[2]:02d}_{use_model}.gif', mode='I', fps=1) as writer:
        for filename in filenames:
            image = imageio.imread(img_dir+filename)
            writer.append_data(image)

def read_omni(filename):
    n = 0
    omni = []
    with open(filename, 'r') as code:
        while True:
            line = code.readline()
            n += 1
            if not line:break 
            array = [eval(i) for i in line.strip().split()]
            omni.append(array)
    return np.array(omni)

def read_one_day_omni(all_omni, DOY, hour):   
    omni_data = all_omni.iloc[(DOY-1)*24+hour].to_list() 
    #print(omni_data)
    #Kp index,R (Sunspot No.),"Dst-index, nT","ap_index, nT",f10.7_index
    #['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.']
    data = list(omni_data[3:])
    return [data[2], data[4], data[0], data[3], data[1], np.nan]

OMNI_2019 = pd.read_csv('omni/2019hourly.csv')
OMNI_2020 = pd.read_csv('omni/2020hourly.csv')
OMNI_2021 = pd.read_csv('omni/2021hourly.csv')
def read_file(filename, map_type='TEC'):
    with open(filename, 'r') as code:
        str_list = []
        tec_for_hour = []
        times, omnis = [], []
        LAT, latitudes = [], []
        n, nn = 1, 0
        while True:
            line = code.readline()
            n += 1
            if not line:break 
            if 'EPOCH OF CURRENT MAP' in line: #record date & time of TEC
                y, m, d, h = [int(d) for d in line.split()[:4]]
                DOY = int(datetime.datetime.strptime(f'{y}-{m}-{d}', '%Y-%m-%d').strftime('%j'))
                if y == 2019:omni = read_one_day_omni(OMNI_2019, DOY, h)
                elif y == 2020:omni = read_one_day_omni(OMNI_2020, DOY, h)
                elif y == 2021:omni = read_one_day_omni(OMNI_2021, DOY, h)
                else: omni = []
                omnis.append(omni)
                #print([y, m, d, h], DOY, omni)
                
                times.append([y, m, d, h])
                
            if 'LAT/LON1/LON2/DLON/H' in line: #record TEC
                tec_data = []
                latitudes.append(eval(line.replace('-180.0', ' ').split()[0])) #add latitude trying to let model know the position

                for _ in range(5):
                    tec_data.append(code.readline().replace('"', '').replace('!', '').replace('\x1a', ''))
                    
                full_data = [int(d) for d in ' '.join(tec_data).split()]    
                tec_for_hour.append(full_data[:-1])
                nn += 1

            if nn == 71:
                nn = 0            
                str_list.append(tec_for_hour)
                LAT.append(latitudes)
                tec_for_hour, latitudes = [], []
        #print(np.array(str_list)[0])
        if map_type=='TEC': return str_list[:24], times[:24], omnis[:24]#, LAT[:25]
        else: return str_list[25:-1], times[25:-1]

def get_rms(rms_map, gim):
    taiwan_point = 25*72 + 61
    rms_local = np.array(rms_map).flatten()[taiwan_point]
    rms_global = np.mean(rms_map)
    max_error, min_error = np.max(rms_map), np.min(rms_map)
    max_tec, min_tec = np.max(gim), np.min(gim)
    return [float("%2.3f"%rms_global), float("%2.3f"%rms_local), float("%2.3f"%max_tec), float("%2.3f"%min_tec), float("%2.3f"%max_error), float("%2.3f"%min_error)]

def plot_tec_rms(target, rms, rms_, omni, DOY, datetimes=[0,0,0,0]):
    rms_, omni = np.expand_dims(rms_, axis=1), np.expand_dims(omni, axis=1)
    print(rms_, omni)
    date_format = f'{datetimes[0]}-{datetimes[1]:02d}-{datetimes[2]:02d} {datetimes[3]:02d}'
    plt.figure(figsize=(12, 6))
    ori_size = plt.rcParams["figure.figsize"]
    print(ori_size)
    plt.figure(figsize=(ori_size[0]*3, ori_size[1]*3))
    plt.rcParams['font.size'] = '30'
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
    ax1.set_title(f'CODE GIM observation')

    cmap = plt.cm.get_cmap('binary')
    ax2 = plt.subplot(222)
    ax2.margins(2, 2)       
    norm = mpl.colors.Normalize(vmin=0, vmax=30) 
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))  
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax2.imshow(rms, cmap = cmap, interpolation='nearest', extent=[-180, 180, -87.5, 87.5], aspect=2)
    ax2.set_title(f'CODE GIM RMS')

    ax3 = plt.subplot(223)
    ax3.axis('off')
    ax3.axis('tight')
    t1 = ax3.table(cellText=omni,
                    colWidths=[0.2]*6,
                    cellLoc='center',
                    rowColours =["palegreen"] * 6,
                    rowLabels=['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.', 'GMS size'],
                    bbox=[0.58, 0.01, 0.2, 1])
    t1.set_fontsize(30)
    ax3.set_title(f'Space weather', x=0.5, fontsize=30)
    
    ax4 = plt.subplot(224)
    ax4.axis('off')
    t2 = ax4.table(cellText=rms_,
                    colWidths=[0.2]*6,
                    cellLoc='center',
                    rowColours=["palegreen"] * 6,
                    rowLabels=['RMSE Global', 'RMSE Taiwan', 'Max TEC', 'Min TEC', 'Max Error', 'Min Error'],
                    bbox=[0.59, 0.01, 0.2, 1])
    t2.set_fontsize(30)
    ax4.set_title(f'Observation and Error', x=0.5, fontsize=30)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.suptitle(f'CODE GIM: {datetimes[0]} Day {DOY:03d} UTC{datetimes[3]:02d}:00', y=0.98, fontsize=50)
    plt.savefig(f'img/observation/{DOY}->{date_format}.png')
    plt.close('all')

def plot_matrix(tec_map):
    matrix = sn.heatmap(tec_map, square=True)#cmap='binary'
    plt.tight_layout() 
    matrix.get_figure().savefig("img/line_chart/matrix.png") 

def get_scatter(data):
    #plt.style.use("ggplot")
    p = sn.color_palette("flare", as_cmap=True)
    scatter = sn.scatterplot(data = data, x='x', y='y', hue='DOY', size='DOY',sizes=(5, 5), palette=p) #, hue=[(60. + i) for i in range]
    scatter.set_xlabel("North Pole")
    scatter.set_ylabel("Sourth Pole")
    scatter.get_figure().savefig("img/line_chart/scatter.png") 


if __name__ == "__main__":
    # read_one_day_omni(all_omni, DOY, hour)
    #x, y = [], []
    days = []
    data = {'x':[], 'y':[], 'DOY':[]}
    omni_path = 'omni/2020hourly.csv'    
    all_omni = pd.read_csv(omni_path)
    file_list = os.listdir('txt/2020')
    for file_name in file_list:
        x, y = [], []
        rms_data, date_rms = read_file('txt/2020/'+file_name, 'RMS')
        tec_data, date_tec, omnis = read_file('txt/2020/'+file_name, 'TEC')
        rms_data, date_rms = np.array(rms_data), np.array(date_rms)
        tec_data, date_tec, omnis = np.array(tec_data), np.array(date_tec), np.array(omnis)
        print(omnis.shape)
        input()
        for idx in range(len(rms_data)):
            
            plot(rms_data[idx], points=None, datetime=date_rms[idx][:], type_='GIM RMS', use_model='')
            input()
            '''
            n_pole = np.mean(tec_data[idx][:11], axis=1)
            s_pole = np.mean(np.flip(tec_data[idx][60:]), axis=1)
            #print(n_pole.dot([]))
            n_pole_w = n_pole * np.cos(np.pi * (np.arange(87.5, 60, -2.5, dtype=float)/180))
            s_pole_w = s_pole * np.cos(np.pi * (np.arange(-87.5, -60, 2.5, dtype=float)/180))
            
            data['x'] += n_pole.tolist()
            data['y'] += s_pole.tolist()
            #plot_matrix(tec_data[idx])
            
            y, m, d, h = date_rms[idx][:]
            DOY, hour = int(datetime.datetime.strptime(f'{y}-{m}-{d}', '%Y-%m-%d').strftime('%j'))+1, int(h)
            days.append(DOY)
            #omni = read_one_day_omni(all_omni, DOY, hour)
            rms_ = get_rms(rms_data[idx], tec_data[idx])
            data['DOY'] += [DOY for _ in range(11)]
            #plot_tec_rms(tec_data[idx], rms_data[idx], rms_, omni, DOY, date_rms[idx])
                      
            #x, y = np.array(x), np.array(y)

            #print(x.shape, y.shape, days)
        print(data['DOY'] )
        input()
    get_scatter(data) 
    '''
        
    '''
    array = read_omni('omni/2019.lst')
    print(array[:, 3:7])
    print(array.shape)
    
    plot_init()
    
    file_list = os.listdir('GIM_CODE/txt/2019')
    for file_name in file_list:
        full_data, datetime = read_file('GIM_CODE/txt/2019/'+file_name)
        full_data = np.array(full_data)
        
        for i in range(full_data.shape[0]):
            plot(full_data[i], datetime[i])
    
    full_data, datetime = read_file('txt/2020/CODG0500.20I')
    full_data = np.array(full_data)
        
    for i in range(full_data.shape[0]):
        plot(full_data[i], datetime[i], type_='Turth')
    generate_gif('img/2020_real/', datetime[0])
    '''