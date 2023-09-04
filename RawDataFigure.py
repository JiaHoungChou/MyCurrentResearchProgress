import os, warnings, copy, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["figure.dpi"]= 300

def DataLengthPreprocessing(data: pd.DataFrame) -> pd.DataFrame:
    raw_data= copy.copy(data)
    if "Time (h)" not in list(raw_data.columns):
        raise ValueError("The input data should contain the Time (h) column")
    raw_data["Time (h)"]= [int(f_element) for f_element in np.floor(raw_data["Time (h)"])]
    raw_data= raw_data.groupby("Time (h)", as_index= False).mean()
    return raw_data

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"

    folder_path= os.path.join(os.getcwd(), "batteryfc2")
    file_ls= os.listdir(folder_path)
    c_name= ["Time (h)", "U1 (V)", "U2 (V)", "U3 (V)", "U4 (V)", 
                     "U5 (V)", "Utot (V)", "J (A/cm2)", "I (A)", "TinH2", 
                     "ToutH2", "TinAIR", "ToutAIR", "TinWAT", "ToutWAT", 
                     "PinAIR (mbara)", "PoutAIR (mbara)", "PoutH2 (mbara)", 
                     "PinH2 (mbara)", "DinH2 (l/mn)", "DoutH2 (l/mn)", "DinAIR (l/mn)", 
                     "DoutAIR (l/mn)", "DWAT (l/mn)", "HrAIRFC (%)"]
    data= pd.DataFrame()
    for i in range(0, len(file_ls)):
        with open(os.path.join(folder_path, file_ls[i]), "r") as file:     
            ls_data= pd.read_csv(os.path.join(folder_path, file_ls[i]), encoding= "Windows-1252")
        ls_data.columns= c_name 
        data= pd.concat([data, ls_data], axis= 0)
    data= data.drop(["U1 (V)", "U2 (V)", "U3 (V)", "U4 (V)", "U5 (V)"], axis= 1)
    data= DataLengthPreprocessing(data)
    
    c_name= list(data.columns)
    plt.figure(figsize= (30, 15))
    for i in range(1, len(c_name)):
        if i== 1:
            plt.subplot(4, 5, i)
            plt.plot(data["Time (h)"], data[c_name[i]], label= c_name[i], color= "darkred")
            plt.grid(True)
            plt.ylabel(c_name[i])
            plt.xlabel("Time (h)")
            plt.legend(loc= "best", fontsize= 18)
        else:
            plt.subplot(4, 5, i)
            plt.plot(data["Time (h)"], data[c_name[i]], label= c_name[i], color= "darkblue")
            plt.ylabel(c_name[i])
            plt.xlabel("Time (h)")
            plt.grid(True)
            plt.legend(loc= "best", fontsize= 18)
    plt.show()
    plt.tight_layout()
    