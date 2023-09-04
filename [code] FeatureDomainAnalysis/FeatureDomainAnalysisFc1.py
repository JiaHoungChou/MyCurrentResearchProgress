import os, warnings, copy, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA
from scipy.optimize import curve_fit
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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
    raw_data["Time (h)"]= [int(f_element) for f_element in np.floor(raw_data["Time (h)"])]
    raw_data= raw_data.groupby("Time (h)", as_index= False).mean()
    return raw_data

def BaconWattsKneepintModel(x, alpha0, alpha1, alpha2, x1):
        return alpha0 + alpha1*(x - x1) + alpha2*(x - x1)*np.tanh((x - x1) / 1e-8)

def DoubleBaconWattsModel(x, alpha0, alpha1, alpha2, alpha3, x0, x2):
    return alpha0 + alpha1*(x - x0) + alpha2*(x - x0)*np.tanh((x - x0)/1e-8) + alpha3*(x - x2)*np.tanh((x - x2)/1e-8)

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

def Interpolation(x, removed_index):
    x= copy.copy(x)
    h= np.arange(0, len(x))
    for i in range(0, len(removed_index)):
        t= removed_index[i]
        if len(x.shape)!= 1:
            x.iloc[t, :]= ((x.iloc[t+ 2, :]- x.iloc[t+ 1, :])/ (h[t+ 2]- h[t+ 1]))* (h[t]- h[t+ 1])+ x.iloc[t+ 1, :]
        else:
            x.iloc[t]= ((x.iloc[t+ 2]- x.iloc[t+ 1])/ (h[t+ 2]- h[t+ 1]))* (h[t]- h[t+ 1])+ x.iloc[t+ 1]
    return x

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"

    folder_path= os.path.join(os.getcwd(), "..", "batteryfc1")
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
    
    show_figure= False
    if show_figure== True:
        c_name= list(data.columns[1: ])
        for i in range(0, len(c_name)):
            plt.figure(figsize= (8, 3))
            if c_name[i]== "Utot (V)":
                plt.plot(data["Time (h)"], data[c_name[i]], label= c_name[i], color= "darkblue")
            else:
                plt.plot(data["Time (h)"], data[c_name[i]], label= c_name[i], color= "black")
            plt.legend(loc= "best", fontsize= 15)
            plt.xticks(fontsize= 15)
            plt.yticks(fontsize= 15)
            plt.xlabel("Time (h)", fontsize= 15)
            plt.ylabel(c_name[i], fontsize= 15)
            plt.grid(True)
            plt.tight_layout()
            # plt.savefig("Variables Curve (fc1) index {}.png".format(str(i+ 1)))
            plt.show()
    
    x, y= data.drop(["Utot (V)", "Time (h)"], axis= 1), data["Utot (V)"]

    polarization_est_h=  [49, 187, 349, 516, 659, 824, 992]
    x= Interpolation(x= x, removed_index= polarization_est_h)
    y= Interpolation(x= y, removed_index= polarization_est_h)

    plt.figure(figsize= (6, 3))
    plt.plot(data["Time (h)"], data["Utot (V)"], color= "darkblue", label= "Vtotal")
    for i in range(0, len(polarization_est_h)):
        if i== len(polarization_est_h)- 1:
            plt.scatter(data["Time (h)"].iloc[polarization_est_h[i]], data["Utot (V)"].iloc[polarization_est_h[i]], edgecolors= "black", facecolor= "none", s= 80, label= "large peak value")
        else:
            plt.scatter(data["Time (h)"].iloc[polarization_est_h[i]], data["Utot (V)"].iloc[polarization_est_h[i]], edgecolors= "black", facecolor= "none", s= 80)
    plt.plot(data["Time (h)"], y, color= "darkred", label= "modified Vtotal")
    plt.legend(loc= "best", fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.xlabel("Time (h)", fontsize= 15)
    plt.ylabel("V", fontsize= 15)
    plt.grid(True)
    plt.show()

    eol_TH= data["Utot (V)"].iloc[0]* 0.96
    eol_H= np.where(np.array(data["Utot (V)"])< eol_TH)[0][0]
    plt.figure(figsize= (6, 3))
    plt.plot(data["Time (h)"], data["Utot (V)"], label= "Vtotal", color= "black")
    plt.scatter(data["Time (h)"].iloc[eol_H], data["Utot (V)"].iloc[eol_H], edgecolors= "black", color= "red", s= 30)
    plt.hlines(eol_TH, data["Time (h)"].iloc[0], data["Time (h)"].iloc[-1], label= "EOL TH (drop to 96%)", color= "darkred", ls= "--")
    plt.legend(loc= "best", fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.xlabel("Time (h)", fontsize= 15)
    plt.ylabel("V", fontsize= 15)
    plt.grid(True)
    plt.show()

    ### data smoothing
    lowess = sm.nonparametric.lowess
    transformation_curve= lowess(y, np.arange(1, len(y)+ 1), frac= 1.0/1.0, it= 0)[:, 1]

    parameter_bound= [1, -1e-4, -1e-4, len(transformation_curve)* 0.7]
    coe, cov= curve_fit(BaconWattsKneepintModel, np.arange(1, len(transformation_curve)+ 1), transformation_curve, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_x1= coe
    upper_x1, lower_x1= coe[3]+ 1.96* np.diag(cov)[3], coe[3]- 1.96* np.diag(cov)[3]
    knee_point= round(opt_x1)
    bacon_watts_first_segment_curve= BaconWattsKneepintModel(np.arange(1, len(transformation_curve)+ 1), opt_alpha0, opt_alpha1, opt_alpha2, opt_x1)[: round(opt_x1)]
    bacon_watts_curve= BaconWattsKneepintModel(np.arange(1, len(transformation_curve)+ 1), opt_alpha0, opt_alpha1, opt_alpha2, opt_x1)
    distance_bacon_watts_degradation= ((bacon_watts_first_segment_curve- transformation_curve[: round(opt_x1)])** 2)
    knee_onset= np.where(distance_bacon_watts_degradation== np.min(distance_bacon_watts_degradation[-500: ]))[0][0]
    parameter_bound= [opt_alpha0, opt_alpha1 + opt_alpha2/2, opt_alpha2, opt_alpha2/2, 0.8*opt_x1, 1.1*opt_x1]
    coe, cov= curve_fit(DoubleBaconWattsModel, np.arange(1, len(transformation_curve)+ 1), transformation_curve, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_alpha3, opt_x0, opt_x2= coe
    upper_x1, lower_x1= coe[4]+ 1.96* np.diag(cov)[4], coe[4]- 1.96* np.diag(cov)[4]
    knee_onset_double_bacon_watts= round(opt_x0)
    double_bacon_watts_curve= DoubleBaconWattsModel(np.arange(1, len(transformation_curve)+ 1), opt_alpha0, opt_alpha1, opt_alpha2, opt_alpha3, opt_x0, opt_x2)

    print("---------------------------------------------------------------------")
    print("The optimal knee point locate at #", knee_point, "cycle")
    print("The optimal knee onset locate at #", knee_onset, "cycle")
    print("The optimal knee onset (double Bacon Watts) locate at #", knee_onset_double_bacon_watts, "cycle")

    plt.figure(figsize= (6, 4))
    plt.plot(data["Time (h)"], y, color= "darkblue", label= "Vtotal")
    plt.plot(data["Time (h)"], transformation_curve, color= "darkgreen", label= "Smoothed Vtotal", lw= 2)
    plt.plot(data["Time (h)"],  bacon_watts_curve, color= "darkred", label= "estimation of Bacon-Watts model", ls= "dotted", lw= 3, alpha= 0.5)
    plt.plot(data["Time (h)"], double_bacon_watts_curve, color= "darkred", label= "estimation of Bacon-Watts model", ls= "dashed", lw= 3, alpha= 0.5)
    plt.scatter(knee_onset_double_bacon_watts,  y[knee_onset_double_bacon_watts], label= "knee onset (Double Bacon-Watts model)", edgecolors= "black", color= "darkred", s= 50, marker= "o")
    plt.scatter(knee_point, y[knee_point], label= "knee point (Bacon-Watts model)", edgecolors= "black", color= "black", s= 50, marker= "d")
    plt.scatter(data["Time (h)"].iloc[eol_H], data["Utot (V)"].iloc[eol_H], label= "EOL", edgecolors= "black", color= "darkblue", s= 50)
    plt.hlines(eol_TH, data["Time (h)"].iloc[0], data["Time (h)"].iloc[-1], label= "EOL TH (drop to 96%)", color= "darkred", ls= "--")
    plt.legend(loc= "best", fontsize= 8)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.xlabel("Time (h)", fontsize= 15)
    plt.ylabel("V", fontsize= 15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    scaler_x, scaler_y= MinMaxScaler(), MinMaxScaler()
    x_feature_selection, y_feature_selection= x.iloc[: knee_point, :], y.iloc[: knee_point]
    c_name= x_feature_selection.columns

    x_feature_selection= scaler_x.fit_transform(x_feature_selection)
    y_feature_selection= scaler_y.fit_transform(np.array(y_feature_selection).reshape(-1, 1)).ravel()

    rf_model= RandomForestRegressor(n_estimators= 150)
    rf_model.fit(x_feature_selection, y_feature_selection)
    
    sorted_idx= rf_model.feature_importances_.argsort()
    plt.figure(figsize= (6, 6))
    plt.barh(c_name[sorted_idx], rf_model.feature_importances_[sorted_idx])
    plt.xlabel("Features", fontsize= 15)
    plt.ylabel("Feature importance score", fontsize= 15)
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize= (6, 6))
    plt.barh(c_name[sorted_idx[::-1]], np.cumsum(rf_model.feature_importances_[sorted_idx[::-1]]))
    plt.xlabel("Features", fontsize= 15)
    plt.ylabel("Feature cumulated importance score", fontsize= 15)
    plt.vlines(0.9, -0.5, len(c_name), color= "red", ls= "--")
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cusum_score= np.cumsum(rf_model.feature_importances_[sorted_idx[::-1]])
    selected_features= list(c_name[sorted_idx[::-1]][: np.where(cusum_score>= 0.80)[0][0]+ 1])

    print("Selected feature name: ", selected_features)
    
    final_x= data[selected_features]
    pca_model_with_feature_selection= PCA(n_components= 2)
    del scaler_x
    scaler_x= MinMaxScaler()
    pca_x_with_feature_selection= scaler_x.fit_transform(final_x.iloc[: knee_point, :])
    pca_model_with_feature_selection.fit(pca_x_with_feature_selection)
    pca_x_with_feature_selection= pca_model_with_feature_selection.transform(pca_x_with_feature_selection)

    del scaler_x
    scaler_x= MinMaxScaler()
    pca_x_without_feature_selection= scaler_x.fit_transform(x.iloc[: knee_point, :])
    
    pca_model_without_feature_selection= PCA(n_components= 2)
    pca_model_without_feature_selection.fit(pca_x_without_feature_selection)
    
    pca_x_without_feature_selection= pca_model_without_feature_selection.transform(pca_x_without_feature_selection)

    plt.figure(figsize= (5, 5))
    plt.scatter(pca_x_with_feature_selection[: knee_point, 0], pca_x_with_feature_selection[: knee_point, 1], edgecolors= "black", color= "darkblue", label= "PCA with feature selection")
    plt.scatter(pca_x_without_feature_selection[: knee_point, 0], pca_x_without_feature_selection[: knee_point, 1], edgecolors= "black", color= "darkred", label= "PCA without feature selection", alpha= 0.5)
    plt.legend(loc= "best", fontsize= 15)
    plt.xlabel("PC1", fontsize= 15)
    plt.ylabel("PC2", fontsize= 15)
    plt.xticks(fontsize= 10)
    plt.yticks(fontsize= 10)
    plt.grid(True)
    plt.show()