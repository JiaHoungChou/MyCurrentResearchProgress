from nni.experiment import Experiment
import os
experiment= Experiment('local')

experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = os.path.join(os.getcwd())
experiment.config.search_space= {
                                                        "batch_size": {"_type": "randint", "_value": [1, 512]},
                                                        "learning_rate": {"_type": "uniform", "_value": [0.0001, 0.1]},
                                                        "dropout_rate": {"_type": "uniform", "_value": [0.01, 0.5]},
                                                        "lstm_layers": {"_type": "choice", "_value": [2, 4, 6]},
                                                        "hidden_size": {"_type": "randint", "_value": [1, 150]},
                                                        "attention_heads": {"_type": "choice", "_value": [2, 3, 4, 5, 6]},
                                                        }

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number= 1000
experiment.config.trial_concurrency= 2
experiment.config.max_trial_duration = "8h"
experiment.run(8080)

pass_code= input("type yes to stop the program and web service")
if pass_code.lower()== "yes":
    experiment.stop()

