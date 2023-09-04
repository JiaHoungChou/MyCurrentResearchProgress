import os, warnings, copy, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from PyEMD import EMD
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["figure.dpi"]= 300
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)

def DataLengthPreprocessing(data: pd.DataFrame) -> pd.DataFrame:
    raw_data= copy.copy(data)
    if "Time (h)" not in list(raw_data.columns):
        raise ValueError("The input data should contain the Time (h) column")
    raw_data["Time (h)"]= [int(f_element) for f_element in np.ceil(raw_data["Time (h)"])]
    raw_data= raw_data.groupby("Time (h)", as_index= False).mean()
    return raw_data

def HampelFilter(x_sequence, k= 32, sigma= 3):
    length= x_sequence.shape[0]- 1
    iLo, iHi= np.array([i- k for i in range(0, length+ 1)]), np.array([i+ k for i in range(0, length+ 1)])
    iLo[iLo < 0], iHi[iHi > length]= 0, length
    xmad= []; xmedian= []
    for i in range(length+ 1):
        w= x_sequence[iLo[i]: iHi[i]+ 1]
        medj= np.median(w)
        mad= np.median(np.abs(w- medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad, xmedian= np.array(xmad), np.array(xmedian)
    scale= 4
    xsigma= scale* xmad
    xi= ~(np.abs(x_sequence- xmedian)<= sigma* xsigma)
    xf= x_sequence.copy()
    xf[xi]= xmedian[xi]
    return xf

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    folder_path_1= os.path.join(os.getcwd(), "batteryfc1")
    folder_path_2= os.path.join(os.getcwd(), "batteryfc2")
    file_ls_1= os.listdir(folder_path_1)
    file_ls_2= os.listdir(folder_path_2)
    
    c_name= ["Time (h)", "U1 (V)", "U2 (V)", "U3 (V)", "U4 (V)", 
                    "U5 (V)", "Utot (V)", "J (A/cm2)", "I (A)", "TinH2", 
                    "ToutH2", "TinAIR", "ToutAIR", "TinWAT", "ToutWAT", 
                    "PinAIR (mbara)", "PoutAIR (mbara)", "PoutH2 (mbara)", 
                    "PinH2 (mbara)", "DinH2 (l/mn)", "DoutH2 (l/mn)", "DinAIR (l/mn)", 
                    "DoutAIR (l/mn)", "DWAT (l/mn)", "HrAIRFC (%)"]

    ### load fc1 dataset
    fc1_data= pd.DataFrame()
    for i in range(0, len(file_ls_1)):
        with open(os.path.join(folder_path_1, file_ls_1[i]), "r") as file:     
            ls_data= pd.read_csv(os.path.join(folder_path_1, file_ls_1[i]), encoding= "Windows-1252")
        ls_data.columns= c_name 
        fc1_data= pd.concat([fc1_data, ls_data], axis= 0)
    fc1_data= fc1_data.drop(["U1 (V)", "U2 (V)", "U3 (V)", "U4 (V)", "U5 (V)"], axis= 1)
    fc1_data= DataLengthPreprocessing(fc1_data)

    ### load fc2 dataset
    fc2_data= pd.DataFrame()
    for i in range(0, len(file_ls_2)):
        with open(os.path.join(folder_path_2, file_ls_2[i]), "r") as file:     
            ls_data= pd.read_csv(os.path.join(folder_path_2, file_ls_2[i]), encoding= "Windows-1252")
        ls_data.columns= c_name 
        fc2_data= pd.concat([fc2_data, ls_data], axis= 0)
    fc2_data= fc2_data.drop(["U1 (V)", "U2 (V)", "U3 (V)", "U4 (V)", "U5 (V)"], axis= 1)
    fc2_data= DataLengthPreprocessing(fc2_data)

    plt.figure(figsize= (6, 3))
    plt.plot(fc1_data["Time (h)"], fc1_data["Utot (V)"], color= "darkblue", label= "FC1 dataset Vtotal")
    plt.plot(fc2_data["Time (h)"], fc2_data["Utot (V)"], color= "darkred", label= "FC2 dataset Vtotal")
    plt.legend(loc= "best", fontsize= 8)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.xlabel("Time (h)", fontsize= 15)
    plt.ylabel("V", fontsize= 15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()