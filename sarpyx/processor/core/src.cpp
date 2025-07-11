import matplotlib
import matplotlib.pyplot as plt
plt.ion()
font={("size"):("5")}
matplotlib.rc("font", **font)
# %% imports
import sys
import time
import pathlib
import numpy as np
import numpy.fft
import pandas as pd
import scipy.ndimage
import scipy.signal
import numpy.polynomial
import numba
import numba.cuda
import cupy as cp
import cupy.fft
# %% echo packet information
if ( pathlib.Path("dfa.csv").is_file() ):
    dfa=pd.read_csv("dfa.csv")
    df=pd.read_csv("df.csv")
    dfc=pd.read_csv("dfc.csv")
else:
    df=pd.read_csv("./o_range.csv")
    dfa=pd.read_csv("./o_all.csv")
    cal_type_desc=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_iso_cal"]
    pol_desc=["txh", "txh_rxh", "txh_rxv", "txh_rxvh", "txv", "txv_rxh", "txv_rxv", "txv_rxvh"]
    rx_desc=["rxv", "rxh"]
    signal_type_desc=["echo", "noise", "na2", "na3", "na4", "na5", "na6", "na7", "tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na13", "na14", "txhiso_cal"]
    # %% calibration packet information
    dfc=pd.read_csv("./o_cal_range.csv")
    dfc["pcc"]=np.mod(dfc.cal_iter, 2)
    dfc["pol_desc"]=list(map(lambda x: pol_desc[x], dfc.pol))
    dfc.pol_desc=dfc.pol_desc.astype("category")
    dfc["rx_desc"]=list(map(lambda x: rx_desc[x], dfc.rx))
    dfc.rx_desc=dfc.rx_desc.astype("category")
    dfc["signal_type_desc"]=list(map(lambda x: signal_type_desc[x], dfc.signal_type))
    dfc.signal_type_desc=dfc.signal_type_desc.astype("category")
    dfc["cal_type_desc"]=list(map(lambda x: cal_type_desc[x], dfc.cal_type))
    dfc.cal_type_desc=dfc.cal_type_desc.astype("category")
    df["pol_desc"]=list(map(lambda x: pol_desc[x], df.pol))
    df.pol_desc=df.pol_desc.astype("category")
    df["rx_desc"]=list(map(lambda x: rx_desc[x], df.rx))
    df.rx_desc=df.rx_desc.astype("category")
    df["signal_type_desc"]=list(map(lambda x: signal_type_desc[x], df.signal_type))
    df.signal_type_desc=df.signal_type_desc.astype("category")
    
    # %% sample rate computation using 3.2.5.4, tab5.1-1, tab5.1-2, txpl and swl description
    decimation_filter_bandwidth=[100, (8.770999908447266e+1), -1, (7.424999999999999e+1), (5.943999862670898e+1), (5.061999893188476e+1), (4.4889999389648433e+1), (2.220000076293945e+1), (5.659000015258788e+1), (4.286000061035156e+1), (1.5100000381469728e+1), (4.834999847412109e+1)]
    decimation_filter_L=[3, 2, -1, 5, 4, 3, 1, 1, 3, 5, 3, 4]
    decimation_filter_M=[4, 3, -1, 9, 9, 8, 3, 6, 7, 16, 26, 11]
    decimation_filter_length_NF=[28, 28, -1, 32, 40, 48, 52, 92, 36, 68, 120, 44]
    decimation_filter_output_offset=[87, 87, -1, 88, 90, 92, 93, 103, 89, 97, 110, 91]
    decimation_filter_swath_desc=["full_bandwidth", "s1_wv1", "n/a", "s2", "s3", "s4", "s5", "ew1", "iw1", "s6_iw3", "ew2_ew3_ew4_ew5", "iw2_wv2"]
    # table 5.1-1 D values as a function of rgdec and C. first index is rgdec
    decimation_filter_D=[[1, 1, 2, 3], [1, 1, 2], [-1], [1, 1, 2, 2, 3, 3, 4, 4, 5], [0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 1, 1, 2, 2, 3, 3], [0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4]]
    dfc["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], dfc.rgdec))
    dfc["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], dfc.rgdec))
    dfc["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], dfc.rgdec))
    dfc["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], dfc.rgdec))
    dfc["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], dfc.rgdec))
    dfc["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], dfc.rgdec))
    df["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], df.rgdec))
    df["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], df.rgdec))
    df["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], df.rgdec))
    df["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], df.rgdec))
    df["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], df.rgdec))
    df["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], df.rgdec))
    dfa["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], dfa.rgdec))
    dfa["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], dfa.rgdec))
    dfa["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], dfa.rgdec))
    dfa["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], dfa.rgdec))
    dfa["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], dfa.rgdec))
    dfa["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], dfa.rgdec))
    print("decimation filter ..")
    fref=(3.7534721374511715e+1)
    dfc["fdec"]=((4)*(fref)*(((dfc.decimation_filter_L)/(dfc.decimation_filter_M))))
    dfc["N3_tx"]=np.ceil(((dfc.fdec)*(dfc.txpl))).astype(np.int)
    dfc["decimation_filter_B"]=((((2)*(dfc.swl)))-(((dfc.decimation_filter_output_offset)+(17))))
    dfc["decimation_filter_C"]=((dfc.decimation_filter_B)-(((dfc.decimation_filter_M)*(((dfc.decimation_filter_B)//(dfc.decimation_filter_M))))))
    dfc["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), dfc.iterrows()))
    df["fdec"]=((4)*(fref)*(((df.decimation_filter_L)/(df.decimation_filter_M))))
    df["N3_tx"]=np.ceil(((df.fdec)*(df.txpl))).astype(np.int)
    df["decimation_filter_B"]=((((2)*(df.swl)))-(((df.decimation_filter_output_offset)+(17))))
    df["decimation_filter_C"]=((df.decimation_filter_B)-(((df.decimation_filter_M)*(((df.decimation_filter_B)//(df.decimation_filter_M))))))
    df["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), df.iterrows()))
    dfa["fdec"]=((4)*(fref)*(((dfa.decimation_filter_L)/(dfa.decimation_filter_M))))
    dfa["N3_tx"]=np.ceil(((dfa.fdec)*(dfa.txpl))).astype(np.int)
    dfa["decimation_filter_B"]=((((2)*(dfa.swl)))-(((dfa.decimation_filter_output_offset)+(17))))
    dfa["decimation_filter_C"]=((dfa.decimation_filter_B)-(((dfa.decimation_filter_M)*(((dfa.decimation_filter_B)//(dfa.decimation_filter_M))))))
    dfa["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), dfa.iterrows()))
    # %% get pulse configuration that is rank pri_counts in the past
    print("pulse from past")
    dfap=dfa.set_index("pri_count")
    dfa["ranked_txprr"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txprr
    dfa["ranked_txprr_"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txprr_
    dfa["ranked_txpl"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpl
    dfa["ranked_txpl_"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpl_
    dfa["ranked_txpsf"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpsf
    dfa["ranked_ses_ssb_tx_pulse_number"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().ses_ssb_tx_pulse_number
    dfa.to_csv("dfa.csv")
    df.to_csv("df.csv")
    dfc.to_csv("dfc.csv")
print("start mmap ..")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", offset=((4)*(2)*(24890)*(10800)), shape=(7400,24890,))
fdec=dfc.iloc[0].fdec
xs_a_us=((cp.arange(ss.shape[1]))/(fdec))
xs_off=((xs_a_us)-((((5.e-1))*(dfc.txpl[0])))-((5.e-1)))
xs_mask=(((((((-5.e-1))*(dfc.txpl[0])))<(xs_off))) & (((xs_off)<((((5.e-1))*(dfc.txpl[0]))))))
arg_nomchirp=((-2)*(np.pi)*(((((xs_off)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*((5.e-1))*(dfc.txprr[0]))))))
z=((xs_mask)*(np.exp(((1j)*(arg_nomchirp)))))
nsig=numba.cuda.to_device(ss[0:2300,:])
csig=cp.asarray(nsig)
cksig=cp.fft.fft(csig, axis=1)
ckz=cp.conj(cp.fft.fft(z))
czsig=cp.fft.ifft(((ckz[cp.newaxis,:])*(cksig)), axis=1)
#%% doppler centroid estimation
phi_accc=cp.angle(cp.sum(((czsig[1:,:])*(cp.conj(czsig[0:-1,:]))), axis=0))
phi_accc_r=scipy.signal.savgol_filter(cp.asnumpy(phi_accc), 101, 1)
plt.plot(cp.asnumpy(phi_accc))
plt.plot(phi_accc_r)