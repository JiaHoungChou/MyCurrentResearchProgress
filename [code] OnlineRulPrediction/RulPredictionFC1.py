import os, warnings, copy, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import LabelEncoder
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

def DataPreprocessing(data, continue_columns, discrete_columns):
    continue_scalers= MinMaxScaler().fit(data[continue_columns[1 :]].values)
    discrete_scalers= {}
    num_classes= []
    for col in discrete_columns:
        srs= data[col].apply(str) 
        discrete_scalers[col]= LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique())
    return continue_scalers, discrete_scalers

def TransformInputs(df, continue_scalers, discrete_scalers, continue_columns, discrete_columns):
    out= df.copy()
    out[continue_columns[1 :]]= continue_scalers.transform(df[continue_columns[1 :]].values)
    for col in discrete_columns:
        string_df= df[col].apply(str)
        out[col]= discrete_scalers[col].transform(string_df)
    return out

def TFT_model_future_prediction_module(train_data, max_encoder_len_param, max_prediction_len_param, best_tft, batch_size, selected_c_name):
    futu_encoder_data= train_data.iloc[-1* max_encoder_len_param: , :]
    futu_decoder_data= pd.DataFrame()
    for c_name in list(futu_encoder_data.columns):
        if c_name== "Time (h)":
            futu_decoder_data[c_name]=[futu_encoder_data["Time (h)"].iloc[-1]+ i for i in range(1, max_prediction_len_param+ 1)]
        elif c_name== "Time_series_id":
            futu_decoder_data[c_name]= ["battery1"]* max_prediction_len_param
        else:
            futu_decoder_data[c_name]= [0]* max_prediction_len_param
    futu_data= pd.concat([futu_encoder_data, futu_decoder_data], axis= 0).reset_index(drop= True)
    futu_encoder_dataloader= TimeSeriesDataSet(
                                                futu_encoder_data,
                                                time_idx= "Time (h)",
                                                target= "Utot (V)",
                                                group_ids= ["Time_series_id"],
                                                max_encoder_length= max_encoder_len_param- max_prediction_len_param,
                                                max_prediction_length= max_prediction_len_param,
                                                static_categoricals=  ["Time_series_id"],
                                                time_varying_known_reals=  ["Time (h)"],
                                                time_varying_unknown_reals= selected_c_name,
                                                add_relative_time_idx= True,
                                                add_target_scales= True,
                                                add_encoder_length= True,
                                                )
    futu_decoder_dataloader= TimeSeriesDataSet.from_dataset(futu_encoder_dataloader, futu_data, predict= True,  stop_randomization= True)
    futu_decoder_model_input= futu_decoder_dataloader.to_dataloader(train= False, batch_size= batch_size, num_workers= 0)
    model_predictions= best_tft.predict(futu_decoder_model_input)
    return model_predictions

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
    original_data= copy.copy(data)
    
    x, y= data.drop(["Utot (V)", "Time (h)"], axis= 1), data["Utot (V)"]

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
    plt.ylabel("Utot (V)", fontsize= 15)
    plt.grid(True)
    plt.show()

    lowess = sm.nonparametric.lowess
    transformation_curve= lowess(y, np.arange(1, len(y)+ 1), frac= 1.0/1.0, it= 0)[:, 1]

    parameter_bound= [1, -1e-4, -1e-4, len(transformation_curve)* 0.7]
    coe, cov= curve_fit(BaconWattsKneepintModel, np.arange(1, len(transformation_curve)+ 1), transformation_curve, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_x1= coe
    upper_x1, lower_x1= coe[3]+ 1.96* np.diag(cov)[3], coe[3]- 1.96* np.diag(cov)[3]
    knee_point= round(opt_x1)
    bacon_watts_first_segment_curve= BaconWattsKneepintModel(np.arange(1, len(transformation_curve)+ 1), opt_alpha0, opt_alpha1, opt_alpha2, opt_x1)[: round(opt_x1)]
    distance_bacon_watts_degradation= ((bacon_watts_first_segment_curve- transformation_curve[: round(opt_x1)])** 2)
    knee_onset= np.where(distance_bacon_watts_degradation== np.min(distance_bacon_watts_degradation[-500: ]))[0][0]
    parameter_bound= [opt_alpha0, opt_alpha1 + opt_alpha2/2, opt_alpha2, opt_alpha2/2, 0.8*opt_x1, 1.1*opt_x1]
    coe, cov= curve_fit(DoubleBaconWattsModel, np.arange(1, len(transformation_curve)+ 1), transformation_curve, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_alpha3, opt_x0, opt_x2= coe
    upper_x1, lower_x1= coe[4]+ 1.96* np.diag(cov)[4], coe[4]- 1.96* np.diag(cov)[4]
    knee_onset_double_bacon_watts= round(opt_x0)

    print("[InFo] prediction parameters")
    print("------------------------------------------------------------------")
    print("[1] The EOL determination: %s h"%(str(eol_H)))
    print("[2] Training length determination: [%s h, %s h]"%(str(knee_onset_double_bacon_watts), str(knee_point)))
    print("[3] The prediction Sp determination: %s h"%(str(knee_point)))
    print("------------------------------------------------------------------")

    selected_c_name= ["DinAIR (l/mn)", "TinWAT", "DoutH2 (l/mn)"]

    data= data.dropna(axis= 1)
    data= data[["Time (h)"]+ selected_c_name+ ["Utot (V)"]]
    original_data= original_data[["Time (h)"]+ selected_c_name+ ["Utot (V)"]]
    data["Time_series_id"] = "FC1"
    data["Time (h)"]= np.arange(1, len(data)+ 1)
    original_data["Time_series_id"] = "FC1"
    original_data["Time (h)"]= np.arange(1, len(original_data)+ 1)
    
    data= data.loc[(data["Time (h)"]>= knee_onset_double_bacon_watts)]
    data= data.loc[(data["Time (h)"]<= knee_point)]
    continue_columns= ["Time (h)"]+ selected_c_name+ ["Utot (V)"]
    continue_columns= ["Time (h)"]+ selected_c_name
    discrete_columns= ["Time_series_id"]
    
    scalar_y= MinMaxScaler()
    scalar_x= MinMaxScaler()
    scalar_y.fit(np.array(data["Utot (V)"]).reshape(-1, 1))
    data.iloc[:, 1: -1]= scalar_x.fit_transform(data.iloc[:, 1: -1])
    data["Time_series_id"]= "battery1"
    original_data.iloc[:, 1: -1]= scalar_x.transform(original_data.iloc[:, 1: -1])
    original_data["Time_series_id"] = "battery1"
    
    ### TFT model encoder length and decoder length hyperparameters.
    max_encoder_length= int(len(data)*0.80)
    max_decoder_length= int(len(data)*0.20)
    
    training_cutoff= data["Time (h)"].max()- max_decoder_length

    training= TimeSeriesDataSet(
                                # data[lambda x: x["Time (h)"]<= training_cutoff],
                                data,
                                time_idx= "Time (h)",
                                target= "Utot (V)",
                                group_ids= ["Time_series_id"],
                                max_encoder_length= max_encoder_length,
                                max_prediction_length= max_decoder_length,
                                static_categoricals=  ["Time_series_id"],
                                time_varying_known_reals=  ["Time (h)"],
                                time_varying_unknown_reals= selected_c_name+  ["Utot (V)"],
                                add_relative_time_idx= True,
                                add_target_scales= True,
                                add_encoder_length= True,
                                )
    
    validation= TimeSeriesDataSet.from_dataset(training, 
                                               data, 
                                               predict= True, 
                                               stop_randomization= True
                                               )
    
    batch_size= 64
    train_dataloader= training.to_dataloader(train= True, batch_size= batch_size, num_workers= 0)
    valid_dataloader= validation.to_dataloader(train= False, batch_size= batch_size, num_workers= 0)

    early_stop_callback= EarlyStopping(monitor= "val_loss", min_delta= 1e-4, patience= 10, verbose= True, mode= "min")
    lr_logger= LearningRateMonitor()
    logger= TensorBoardLogger("lightning_logs") 
    quantiles= [0.1, 0.5, 0.9]

    pl.seed_everything(123)
    trainer= pl.Trainer(
                        max_epochs= 1000,
                        accelerator= "auto",
                        devices= 1,
                        enable_model_summary= True,
                        gradient_clip_val= 0.7095432127294071,
                        limit_train_batches= 30,
                        log_every_n_steps= 10,
                        callbacks=[lr_logger, early_stop_callback],
                        logger= logger
                        )
    
    tft= TemporalFusionTransformer.from_dataset(
                                                training,
                                                learning_rate= 0.001,
                                                hidden_size= 150,
                                                attention_head_size= 6,
                                                lstm_layers= 4,
                                                dropout= 0.001,
                                                hidden_continuous_size= 150,
                                                loss= QuantileLoss(quantiles= quantiles),
                                                output_size= len(quantiles),
                                                log_interval= 2, 
                                                reduce_on_plateau_patience= 4,
                                                optimizer= "adam",
                                                encoder_and_decoder_model= "BiLSTM"
                                                )

    trainer.fit(tft, train_dataloaders= train_dataloader, val_dataloaders= valid_dataloader)
    
    best_model_path= trainer.checkpoint_callback.best_model_path
    best_tft= TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    val_predictions= best_tft.predict(valid_dataloader, mode= "raw", return_x= True)
    val_predictions_y= scalar_y.inverse_transform(val_predictions.output.prediction.data.cpu().numpy()[:, :, 1].ravel().reshape(-1, 1))
    val_actuals=  data["Utot (V)"].iloc[-1* len(val_predictions.output.prediction.data.cpu().numpy()[:, :, 1].ravel()) :]
    val_actuals= scalar_y.inverse_transform(np.array(val_actuals).reshape(-1, 1)).ravel()
    
    plt.figure(figsize= (6, 3))
    plt.plot(data["Time (h)"], scalar_y.inverse_transform(np.array(data["Utot (V)"]).reshape(-1, 1)),  color= "darkblue")
    plt.plot(val_predictions.x["decoder_time_idx"].data.cpu().numpy().ravel(), val_actuals, label= "actual", color= "darkblue")
    plt.plot(val_predictions.x["decoder_time_idx"].data.cpu().numpy().ravel(), val_predictions_y, label= "prediction", color= "red")
    plt.legend(loc= "best", fontsize= 12)
    plt.grid(True)
    plt.xlabel("Time(h)", fontsize= 18)
    plt.ylabel("V", fontsize= 18)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.show()
    
    print("-------------------------------------------------------------------------------------------------------------------")
    mae= np.mean(np.abs(val_actuals- val_predictions.output.prediction.data.cpu().numpy()[:, :, 1].ravel()))
    print("[validation performance] mae: {:.4f}".format(mae))
    mse= np.mean((val_actuals- val_predictions.output.prediction.data.cpu().numpy()[:, :, 1].ravel())** 2)
    print("[validation performance] mse: {:.4f}".format(mse))

    ### feature importance
    # interpretation= best_tft.interpret_output(val_predictions.output, reduction="sum")
    # best_tft.plot_interpretation(interpretation)

    ### testing data using future prediction
    selected_c_name= selected_c_name+ ["Utot (V)"]
    max_encoder_length= 2
    max_decoder_length= 1
    model_future_predictions= TFT_model_future_prediction_module(train_data= data, 
                                                                                                                max_encoder_len_param= max_encoder_length, 
                                                                                                                max_prediction_len_param= max_decoder_length,
                                                                                                                best_tft= best_tft, 
                                                                                                                batch_size= batch_size,
                                                                                                                selected_c_name= selected_c_name
                                                                                                                )
    model_future_predictions= scalar_y.inverse_transform(model_future_predictions.data.cpu().numpy().reshape(-1, 1)).ravel()
    model_future_predictions= list(model_future_predictions)
    ### online rolling prediction
    max_prediction_step= 100000
    index= [data.index[-1], data.index[-1]+ max_encoder_length]
    for i in range(0, max_prediction_step):
        known_input= original_data.iloc[index[0]: index[1], :]
        rolling_model_future_predictions= TFT_model_future_prediction_module(train_data= known_input, 
                                                                                                                                 max_encoder_len_param= max_encoder_length, 
                                                                                                                                 max_prediction_len_param= max_decoder_length,  
                                                                                                                                 best_tft= best_tft,
                                                                                                                                 batch_size= batch_size,
                                                                                                                                 selected_c_name= selected_c_name
                                                                                                                                 )
        rolling_model_future_predictions= scalar_y.inverse_transform(rolling_model_future_predictions.data.cpu().numpy().reshape(-1, 1)).ravel()
        model_future_predictions+= list(rolling_model_future_predictions)
        if len(np.where(np.array(model_future_predictions)<= eol_TH)[0]) != 0:
            model_future_predictions= np.array(model_future_predictions)
            if len(np.where(model_future_predictions<= eol_TH)[0])> 1:
                    model_future_predictions= model_future_predictions[: np.where(model_future_predictions<= eol_TH)[0][1]]
            actual_rul= eol_H-  data["Time (h)"].iloc[-1]
            predicted_rul= len(model_future_predictions)
            print("Actu RUL: %d// Pred RUL: %d"%(actual_rul, predicted_rul))
            print("AE: %d"%(np.abs(actual_rul- predicted_rul)))
            print("RE: %2.4f"%(round(np.abs(actual_rul- predicted_rul)/ actual_rul* 100, 4)))
            break
        else:
            index[0]+= max_decoder_length
            index[1]+= max_decoder_length
    
    original_data.iloc[:, 1: -1]= scalar_x.inverse_transform(original_data.iloc[:, 1: -1])
    unknow_data= original_data.iloc[np.where(np.array(original_data["Time (h)"])> data["Time (h)"].iloc[-1] )[0][0]: , :]
    plt.figure(figsize= (6, 3))
    plt.plot(original_data["Time (h)"].iloc[: unknow_data.index[0]], original_data["Utot (V)"].iloc[: unknow_data.index[0]], label= "known past data", color= "black")
    plt.plot(unknow_data["Time (h)"], unknow_data["Utot (V)"], label= "unknown future data", color= "darkblue", alpha= 0.7)
    plt.plot(np.arange(unknow_data.index[0], unknow_data.index[0]+ len(model_future_predictions)), model_future_predictions, label= "prediction", color= "darkred")
    plt.scatter(original_data["Time (h)"].iloc[knee_point], original_data["Utot (V)"].iloc[knee_point], label= "initial Sp", color= "darkgreen")
    plt.scatter(original_data["Time (h)"].iloc[eol_H], original_data["Utot (V)"].iloc[eol_H], label= "determined EOL", color= "red")
    plt.hlines(eol_TH, original_data["Time (h)"].iloc[0], original_data["Time (h)"].iloc[-1], label= "EOL", color= "red", ls= "--")
    plt.legend(loc= "best", fontsize= 10)
    plt.grid(True)
    plt.xlabel("Time (h)", fontsize= 18)
    plt.ylabel("V", fontsize= 18)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.show()

    ### mutiple steps prediction experiments.
    max_encoder_length_ls= [2 , 4, 6, 8, 10,]
    max_decoder_length_ls= [  1,   3,   5,   7,   9,]
    prediction_results_ls= []
    predicted_rul_ls= []
    AE_ls= []
    RE_ls= []
    original_data.iloc[:, 1: -1]= scalar_x.transform(original_data.iloc[:, 1: -1])
    for i in range(0, len(max_encoder_length_ls)):
        max_encoder_length= max_encoder_length_ls[i]
        max_decoder_length= max_decoder_length_ls[i]
        model_future_predictions= TFT_model_future_prediction_module(train_data= data, 
                                                                                                                    max_encoder_len_param= max_encoder_length, 
                                                                                                                    max_prediction_len_param= max_decoder_length,
                                                                                                                    best_tft= best_tft, 
                                                                                                                    batch_size= batch_size,
                                                                                                                    selected_c_name= selected_c_name
                                                                                                                    )
        model_future_predictions= scalar_y.inverse_transform(model_future_predictions.data.cpu().numpy().reshape(-1, 1)).ravel()
        model_future_predictions= list(model_future_predictions)
        ### online rolling prediction
        max_prediction_step= 100000
        index= [data.index[-1], data.index[-1]+ max_encoder_length]
        for j in range(0, max_prediction_step):
            known_input= original_data.iloc[index[0]: index[1], :]
            rolling_model_future_predictions= TFT_model_future_prediction_module(train_data= known_input, 
                                                                                                                                    max_encoder_len_param= max_encoder_length, 
                                                                                                                                    max_prediction_len_param= max_decoder_length,  
                                                                                                                                    best_tft= best_tft,
                                                                                                                                    batch_size= batch_size,
                                                                                                                                    selected_c_name= selected_c_name
                                                                                                                                    )
            rolling_model_future_predictions= scalar_y.inverse_transform(rolling_model_future_predictions.data.cpu().numpy().reshape(-1, 1)).ravel()
            model_future_predictions+= list(rolling_model_future_predictions)
            if len(np.where(np.array(model_future_predictions)<= eol_TH)[0]) != 0:
                model_future_predictions= np.array(model_future_predictions)
                if len(np.where(model_future_predictions<= eol_TH)[0])> 1:
                    model_future_predictions= model_future_predictions[: np.where(model_future_predictions<= eol_TH)[0][1]]
                actual_rul= eol_H-  data["Time (h)"].iloc[-1]
                predicted_rul= len(model_future_predictions)

                prediction_results_ls.append(model_future_predictions)
                predicted_rul_ls.append(predicted_rul)
                AE_ls.append(np.abs(actual_rul- predicted_rul))
                RE_ls.append(np.abs(actual_rul- predicted_rul)/ actual_rul* 100)
                break
            else:
                index[0]+= max_decoder_length
                index[1]+= max_decoder_length
    
    original_data.iloc[:, 1: -1]= scalar_x.inverse_transform(original_data.iloc[:, 1: -1])
    unknow_data= original_data.iloc[np.where(np.array(original_data["Time (h)"])> data["Time (h)"].iloc[-1] )[0][0]: , :]
    plt.figure(figsize= (8, 4))
    plt.plot(original_data["Time (h)"].iloc[: unknow_data.index[0]], original_data["Utot (V)"].iloc[: unknow_data.index[0]], label= "known past data", color= "black")
    plt.plot(unknow_data["Time (h)"], unknow_data["Utot (V)"], label= "unknown future data", color= "darkblue", alpha= 0.7)
    for i in range(0, len(max_decoder_length_ls),):
        plt.plot(np.arange(unknow_data.index[0], unknow_data.index[0]+ len(prediction_results_ls[i])), prediction_results_ls[i], label= "prediction step: %s"%(str(max_decoder_length_ls[i])), alpha= 0.7)
    plt.scatter(original_data["Time (h)"].iloc[knee_point], original_data["Utot (V)"].iloc[knee_point], label= "initial Sp", color= "darkgreen")
    plt.scatter(original_data["Time (h)"].iloc[eol_H], original_data["Utot (V)"].iloc[eol_H], label= "determined EOL", color= "red")
    plt.hlines(eol_TH, original_data["Time (h)"].iloc[0], original_data["Time (h)"].iloc[-1], label= "EOL", color= "red", ls= "--")
    plt.legend(loc= "upper left", fontsize= 12)
    plt.grid(True)
    plt.xlabel("Time (h)", fontsize= 18)
    plt.ylabel("V", fontsize= 18)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.show()

    plt.figure(figsize= (6, 3))
    for i in range(0, len(max_decoder_length_ls)):
        if i== 0:
            plt.scatter(max_decoder_length_ls[i],  AE_ls[i],  label= "AE", color= "black")
        else:
            plt.scatter(max_decoder_length_ls[i],  AE_ls[i], color= "black")
    plt.plot(max_decoder_length_ls,  AE_ls, color= "black")
    plt.legend(loc= "best", fontsize= 12)
    plt.grid(True)
    plt.xlabel("Prediction steps", fontsize= 18)
    plt.ylabel("AE", fontsize= 18)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.show()

    plt.figure(figsize= (6, 3))
    for i in range(0, len(max_decoder_length_ls)):
        if i== 0:
            plt.scatter(max_decoder_length_ls[i],  RE_ls[i],  label= "RE", color= "black")
        else:
            plt.scatter(max_decoder_length_ls[i],  RE_ls[i], color= "black")
    plt.plot(max_decoder_length_ls,  RE_ls, color= "black")
    plt.legend(loc= "best", fontsize= 12)
    plt.grid(True)
    plt.xlabel("Prediction steps", fontsize= 18)
    plt.ylabel("RE", fontsize= 18)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize= 16)
    plt.show()

