import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, os, copy, warnings, nni
import statsmodels.api as sm
import lightning.pytorch as pl
from scipy.optimize import curve_fit
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
plt.rcParams["font.family"]= "Times New Roman"
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

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"

    params= {"batch_size": 210, "learning_rate": 0.08111701554918789,  "dropout_rate": 0.42922764080589926, "lstm_layers": 6, "hidden_size": 118, "attention_heads": 3}

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    folder_path= os.path.join(os.getcwd(), "..", "..", "batteryfc1")
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

    ### create dataloaders for  our model
    batch_size= int(params["batch_size"])
    ### if you have a strong GPU, feel free to increase the number of workers  
    train_dataloader= training.to_dataloader(train= True, batch_size= batch_size, num_workers= 0)
    val_dataloader= validation.to_dataloader(train= False, batch_size= batch_size, num_workers= 0)
    ### The training.index and validation.index show how our training and validation instances are sliced. ##############
    ### define callbacks
    early_stop_callback= EarlyStopping(monitor= "val_loss", min_delta= 1e-4, patience= 10, verbose= True, mode= "min")
    lr_logger= LearningRateMonitor()  
    logger= TensorBoardLogger("lightning_logs") 
    quantiles= [0.1, 0.5, 0.9]
    ### create trainer
    pl.seed_everything(42)
    trainer= pl.Trainer(
                                  max_epochs= 1000,
                                 #weights_summary="top",
                                 accelerator= 'auto',
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
                                                                                learning_rate= float(params["learning_rate"]),
                                                                                hidden_size= int(params["hidden_size"]),
                                                                                attention_head_size= int(params["attention_heads"]),
                                                                                lstm_layers= int(params["lstm_layers"]),
                                                                                dropout= float(params["dropout_rate"]),
                                                                                hidden_continuous_size= 30,
                                                                                loss= QuantileLoss(quantiles=quantiles),
                                                                                output_size= len(quantiles),
                                                                                log_interval= 2, 
                                                                                reduce_on_plateau_patience= 4,
                                                                                optimizer= "adam",
                                                                                encoder_and_decoder_model= "BiLSTM"
                                                                                )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    ######################## model training #######################
    trainer.fit(tft, train_dataloaders= train_dataloader, val_dataloaders= val_dataloader)
    
    best_model_path= trainer.checkpoint_callback.best_model_path
    best_tft= TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    actuals= torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions= best_tft.predict(val_dataloader)

    mse= mean_squared_error(actuals, predictions.data.cpu())
    nni.report_final_result(mse)