from ooipy.hydrophone.basic import Spectrogram
import ooipy
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def get_spdf(spec_like, fs_hz, fmax=None, spl_bins=np.linspace(0, 120, 481),
             percentiles=[1, 5, 10, 50, 90, 95, 99]):
    """
    spec_like : Spectrogram like object. Accaptable are ooipy.Spectrogram or a dictionary
        with keys "time", "freq", and "values"
    fs_hz : sampling frequency in Hz
    fmax : frequency up to which spectral PDF is computed
    spl_bins : bins for spectral level
    percentiles : percentiles that will be computed along with the PDF
    """
    
    if isinstance(spec_like, Spectrogram):
        time = spec_like.time
        freq = spec_like.freq
        values = spec_like.values
    elif isinstance(spec_like, dict):
        try:
            time = spec_like['time']
            freq = spec_like['freq']
            values = spec_like['values']
        except:
            print('spec_like must be either ooipy.Spectrogram object or dictionary with keys "time", "freq", and "values"')
            
    else:
        print('spec_like must be either ooipy.Spectrogram object or dictionary with keys "time", "freq", and "values"')
        
    
    
    dct_category = {'data': [], 'cnt': 0, 'starttime': [], 'endtime': [], 'windspeed': [], 'windangle': []}
    
    if fmax is None:
        fmax = freq[-1]
    
    n_freq_bin = int(len(freq) * fmax/(fs_hz/2)) + 1
       
    spdf_dct = {'freq': np.array(np.linspace(0, fmax, n_freq_bin)),
                'spl': spl_bins[:-1],
                'pdf': np.empty((n_freq_bin, 480)),
                'mean': np.empty(n_freq_bin),
                'number_psd': len(time)}
    
    for p in percentiles:
        spdf_dct[str(p)] = np.empty(n_freq_bin)             

    for idx, freq_bin in enumerate(tqdm(values.transpose()[:n_freq_bin - 1])):
        hist, bin_edges = np.histogram(freq_bin, bins=spl_bins, density=True)
        spdf_dct['pdf'][idx] = hist
        spdf_dct['50'][idx] = np.median(freq_bin)
        spdf_dct['mean'][idx] = np.mean(freq_bin)
        for p in percentiles:
            spdf_dct[str(p)][idx] = np.nanquantile(freq_bin, p/100)

    return spdf_dct


def plot_spdf(spdf, vmin=0.003, vmax=0.2, vdelta=0.0025, save=False, filename=None, log=True, title='Spectral PDF'):
    #plotting spectral pdf and some percentiles:
    #vmin = 0.003 # min probability density
    #vmax = 0.20 # max probability density
    #vdelta = 0.0025
    cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
    fig, ax = plt.subplots(figsize=(16,9))
    im = ax.contourf(spdf['freq'], spdf['spl'], np.transpose(spdf['pdf']),
                     cbarticks, norm=colors.Normalize(vmin=vmin, vmax=vmax),
                     cmap='jet', extend='max', alpha=0.50)

    # plot some percentiles:
    1, 5, 10, 50, 90, 95, 99
    plt.plot(spdf['freq'], spdf['1'], color='black')
    plt.plot(spdf['freq'], spdf['5'], color='black')
    plt.plot(spdf['freq'], spdf['10'], color='black')
    plt.plot(spdf['freq'], spdf['50'], color='black')
    plt.plot(spdf['freq'], spdf['90'], color='black')
    plt.plot(spdf['freq'], spdf['95'], color='black')
    plt.plot(spdf['freq'], spdf['99'], color='black')
    
    plt.ylabel(r'spectral level (dB rel $1 \frac{\mu Pa^2}{Hz}$)')
    plt.xlabel('frequency (Hz)')
    plt.ylim([40,110])
    plt.xlim([0.1, 100])
    if log: plt.xscale('log')
    plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax+vdelta, 0.05),  pad=0.1, label='probability')
    plt.tick_params(axis='y')
    plt.grid(True)
    plt.title(title)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label = 'percentiles: 1, 5, 10, 50, 90, 95, 99',color='k')
    handles.extend([line])
    plt.legend(handles=handles)
    
    if save:
        fig.savefig(filename, dpi=200)
        plt.close()
    else:
        plt.show()

def get_hdatas(i,isolated_ships_2,group_var,hdatas):
    
    vessel_type=isolated_ships_2[group_var][i]
    starttime = isolated_ships_2.start_time[i] 
    endtime = isolated_ships_2.end_time[i]  
    hydro=isolated_ships_2.hydrophone[i] 
    if hydro==1:
        node='Axial_Base'
        hdata = ooipy.request.hydrophone_request.get_acoustic_data_LF(starttime, endtime,node )
    if hydro==2:
        node='AXCC1'
        hdata = ooipy.request.hydrophone_request.get_acoustic_data_LF(starttime, endtime,node )
    if hydro==3:
        node='AXEC2'
        hdata = ooipy.request.hydrophone_request.get_acoustic_data_LF(starttime, endtime,node )
        
    if vessel_type not in hdatas.keys():
        hdatas[vessel_type]=[]
        hdatas[vessel_type].append(hdata)     
    else:
        hdatas[vessel_type].append(hdata)
    return hdatas


def get_psds():
    psds = dict()
    for vessel_type,hdatas_lst in hdatas.items():
        for hdata in hdatas_lst:
            if hdata is None:
                continue
            psd = hdata.compute_psd_welch(L=1024)
            if vessel_type not in psds.keys():
                psds[vessel_type]=[]
                psds[vessel_type].append(psd)     
            else:
                psds[vessel_type].append(psd)
                
def plot_and_save_spdfs(isolated_ships_2,group_var,save=False,plot=True,title=None):
    
    hdatas = dict()
    ## Get the hdatas
    start_exe=time.time()
    results = Parallel(n_jobs=2)(delayed(get_hdatas)(i,isolated_ships_2,group_var,hdatas) for i in tqdm(range(len(isolated_ships_2))) )
    end_exe=time.time()
    print(end_exe-start_exe)
    
    ## Calculate PSD for each time segment
    psds = dict()
    i=0
    for result in results:

        hdata=list(result.values())[0][0]
        vessel_type=list(result.keys())[0]
        if hdata is None:
            continue
        else:
            try:
                psd = hdata.compute_psd_welch(L=1024)
            except:
                print('Time duration is too short')
                print(isolated_ships_2['start_time'].iloc[i])
                print(isolated_ships_2['end_time'].iloc[i])
                pass
            if vessel_type not in psds.keys():
                psds[vessel_type]=[]
                psds[vessel_type].append(psd)     
            else:
                psds[vessel_type].append(psd)
        i=i+1
        
        ## Concatenate all PSDs into single numpy array
        # first create a list of 1D np arrays (not ooipy.PSD objects)
        psds_dict = dict()
        for vessel_type,psds_lst in psds.items():
            for psd in psds_lst:
                if vessel_type not in psds_dict.keys():
                    psds_dict[vessel_type]=[]
                    psds_dict[vessel_type].append(psd.values)
                else:
                    psds_dict[vessel_type].append(psd.values)   

        for vessel_type in psds_dict.keys():
            psds_dict[vessel_type]=np.array(psds_dict[vessel_type])
            
        if save:
            np.save('json_files/{}.npy'.format(title),psds_dict)
        
        if plot:
            ## Calculate and Plot SPDFs
            for k,v in psds_dict.items():
                psd_dict = {
                'values':v,
                'freq':psds[k][0].freq,
                'time':np.arange(v.shape[0])}
                spdf = get_spdf(psd_dict, 200, fmax=100, spl_bins=np.linspace(0, 120, 481),percentiles=[1, 5, 10, 50, 90, 95, 99])
                print(k)
                plot_spdf(spdf, log=False)