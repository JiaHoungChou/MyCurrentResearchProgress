import torch.nn as nn
import torch
import torch.nn.functional as F

class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size, batch_first, trainable):
        super().__init__()
        self.output_size= output_size
        self.batch_first= batch_first
        self.trainable= trainable
        if self.trainable:
            self.mask= nn.Parameter(torch.zeros(self.output_size, dtype= torch.float32))
            self.gate= nn.Sigmoid()
    def interpolate(self, x):
        ### 利用內插法進行上/下採樣
        upsampled= F.interpolate(x.unsqueeze(1), self.output_size, mode= "linear", align_corners= True).squeeze(1)
        if self.trainable:
            upsampled= upsampled* self.gate(self.mask.unsqueeze(0))* 2.0
        return upsampled
    def forward(self, x):
        if len(x.size())<= 2:
            return self.interpolate(x)
        ### (samples* timesteps, input_size)
        x_reshape= x.contiguous().view(-1, x.size(-1))
        y= self.interpolate(x_reshape)
        if self.batch_first:
            ### (samples, timesteps, output_size)
            y= y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            ### (timesteps, samples, output_size)
            y= y.view(-1, x.size(1), y.size(-1))  
        return y

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size* 2)
        self.init_weights()
    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x
    
class ResampleNorm(nn.Module):
    def __init__(self, input_size, output_size, trainable_add):
        super().__init__()
        self.input_size= input_size
        self.trainable_add= trainable_add
        self.output_size= output_size or input_size
        if self.input_size != self.output_size:
            self.resample= TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)
    def forward(self, x: torch.Tensor):
        if self.input_size != self.output_size:
            x= self.resample(x)
        if self.trainable_add:
            x= x * self.gate(self.mask)* 2.0
        output= self.norm(x)
        return output

class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)
    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip= self.resample(skip)
        if self.trainable_add:
            skip= skip * self.gate(self.mask) * 2.0
        output= self.norm(x + skip)
        return output

class GateAddNorm(nn.Module):
    def __init__(self, input_size, hidden_size, skip_size, trainable_add, dropout):
        super().__init__()
        self.input_size= input_size
        self.hidden_size= hidden_size or input_size
        self.skip_size= skip_size or self.hidden_size
        self.dropout = dropout
        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)
    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output
       
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, context_size, residual: bool):
        super().__init__()
        self.input_size= input_size
        self.output_size= output_size
        self.context_size= context_size
        self.hidden_size= hidden_size
        self.dropout= dropout
        self.residual= residual
        ### The shape between the output size and residual size should be the same.
        if self.input_size != self.output_size and not self.residual:
            residual_size= self.input_size
        else:
            residual_size= self.output_size
        if self.output_size != residual_size:
            self.resample_norm= ResampleNorm(residual_size, self.output_size)
        self.fc1= nn.Linear(self.input_size, self.hidden_size)
        self.elu= nn.ELU()
        if self.context_size is not None:
            self.context= nn.Linear(self.context_size, self.hidden_size, bias=False)
        self.fc2= nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()
        self.gate_norm= GateAddNorm(input_size= self.hidden_size, skip_size= self.output_size, hidden_size= self.output_size, dropout= self.dropout, trainable_add= False)
    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)
    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual= x
        if self.input_size != self.output_size and not self.residual:
            residual= self.resample_norm(residual)
        x= self.fc1(x)
        if context is not None:
            context= self.context(context)
            x= x + context
        x= self.elu(x)
        x= self.fc2(x)
        x= self.gate_norm(x, residual)
        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, input_embedding_flags, dropout, context_size, single_variable_grns, prescalers):
        super().__init__()
        self.hidden_size= hidden_size
        self.input_sizes= input_sizes
        self.input_embedding_flags= input_embedding_flags
        self.dropout= dropout
        self.context_size= context_size
        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn= GatedResidualNetwork(self.input_size_total, min(self.hidden_size, self.num_inputs), self.num_inputs, self.dropout, self.context_size, residual= False)
            else:
                self.flattened_grn= GatedResidualNetwork(self.input_size_total, min(self.hidden_size, self.num_inputs), self.num_inputs, self.dropout, residual= False)
        self.single_variable_grns= nn.ModuleDict()
        self.prescalers= nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name]= single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name]= ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name]= GatedResidualNetwork(input_size, min(input_size, self.hidden_size), output_size= self.hidden_size, dropout= self.dropout)
            if name in prescalers:
                self.prescalers[name]= prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name]= nn.Linear(1, input_size)
        self.softmax = nn.Softmax(dim=-1)
    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())
    @property
    def num_inputs(self):
        return len(self.input_sizes)
    def forward(self, x, context):
        if self.num_inputs > 1:
            ### transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                ### select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding= self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim= -1)
            ### calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)
            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:  
            ### for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  
                ### -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  
                ### ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparse_weights
    
class TemporalFusionTransformer(nn.Module):
    def __init__(self, hidden_size, lstm_layers, dropout, output_size, attention_head_size, max_encoder_length, static_categoricals, 
                        static_reals, time_varying_categoricals_encoder, time_varying_categoricals_decoder, categorical_groups,
                        time_varying_reals_encoder, time_varying_reals_decoder, x_reals, x_categoricals, hidden_continuous_size,
                        hidden_continuous_sizes, embedding_sizes, embedding_paddings, embedding_labels, learning_rate,
                        log_interval, log_val_interval, log_gradient_flow, reduce_on_plateau_patience, monotone_constaints,
                        share_single_variable_networks, logging_metrics, encoder_and_decoder_model_name,
                        **kwargs):
        super().__init__(**kwargs)
        self.encoder_and_decoder_model_name= encoder_and_decoder_model_name
        self.prescalers= nn.ModuleDict({
            name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
            for name in self.reals
            }
            )
        static_input_sizes= {
            name: self.input_embeddings.output_size[name] for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes= static_input_sizes,
            hidden_size= self.hparams.hidden_size,
            input_embedding_flags= {name: True for name in self.hparams.static_categoricals},
            dropout= self.hparams.dropout,
            prescalers= self.prescalers,
        )
        ### variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )
        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )
        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )
        
        if self.encoder_and_decoder_model== None or self.encoder_and_decoder_model== "LSTM":
            # lstm encoder (history) and decoder (future) for local processing
            self.encoder_model= LSTM(
                                    input_size=self.hparams.hidden_size,
                                    hidden_size=self.hparams.hidden_size,
                                    num_layers=self.hparams.lstm_layers,
                                    dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                    batch_first=True,
                                    )
            self.decoder_model= LSTM(
                                    input_size=self.hparams.hidden_size,
                                    hidden_size=self.hparams.hidden_size,
                                    num_layers=self.hparams.lstm_layers,
                                    dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                    batch_first=True,
                                    )
        elif self.encoder_and_decoder_model== "BiLSTM":
            self.encoder_model= LSTM(
                                    input_size=self.hparams.hidden_size,
                                    hidden_size=self.hparams.hidden_size,
                                    num_layers=self.hparams.lstm_layers,
                                    dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                    batch_first=True,
                                    bidirectional= True
                                    )
            self.decoder_model= LSTM(
                                    input_size=self.hparams.hidden_size,
                                    hidden_size=self.hparams.hidden_size,
                                    num_layers=self.hparams.lstm_layers,
                                    dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                    batch_first=True,
                                    bidirectional= True
                                    )
            self.BiLSTM_dense_layer_encoder= nn.Linear(self.hparams.hidden_size* 2, self.hparams.hidden_size)
            self.BiLSTM_dense_layer_decoder= nn.Linear(self.hparams.hidden_size* 2, self.hparams.hidden_size)
        
        elif self.encoder_and_decoder_model== "GRU":
            self.encoder_model= nn.GRU(
                                       input_size=self.hparams.hidden_size,
                                       hidden_size=self.hparams.hidden_size,
                                       num_layers=self.hparams.lstm_layers,
                                       dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                       batch_first=True,
                                       ) 
            self.decoder_model= nn.GRU(
                                       input_size=self.hparams.hidden_size,
                                       hidden_size=self.hparams.hidden_size,
                                       num_layers=self.hparams.lstm_layers,
                                       dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                       batch_first=True,
                                       )
        elif self.encoder_and_decoder_model== "BiGRU":
            self.encoder_model= nn.GRU(
                                       input_size=self.hparams.hidden_size,
                                       hidden_size=self.hparams.hidden_size,
                                       num_layers=self.hparams.lstm_layers,
                                       dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                       batch_first=True,
                                       bidirectional= True
                                      )
            self.decoder_model= nn.GRU(
                                       input_size=self.hparams.hidden_size,
                                       hidden_size=self.hparams.hidden_size,
                                       num_layers=self.hparams.lstm_layers,
                                       dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
                                       batch_first=True,
                                       bidirectional= True
                                       )
            self.GRU_dense_layer_encoder= nn.Linear(self.hparams.hidden_size* 2, self.hparams.hidden_size)
            self.GRU_dense_layer_decoder= nn.Linear(self.hparams.hidden_size* 2, self.hparams.hidden_size)
            
            
           

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)

        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, output_size) for output_size in self.hparams.output_size]
            )
        else:
            self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        # update defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )
        if self.encoder_and_decoder_model== None or self.encoder_and_decoder_model== "LSTM":
            # LSTM
            # calculate initial state
            input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)
            input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)
            # run local encoder
            encoder_output, (hidden, cell)= self.encoder_model(embeddings_varying_encoder, (input_hidden, input_cell), lengths= encoder_lengths, enforce_sorted= False)
            # run local decoder
            decoder_output, _= self.decoder_model(embeddings_varying_decoder, (hidden, cell), lengths= decoder_lengths, enforce_sorted= False)
        elif self.encoder_and_decoder_model== "BiLSTM":
            # BiLSTM
            # calculate initial state
            input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(self.hparams.lstm_layers* 2, -1, -1)
            input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers* 2, -1, -1)
            encoder_output, (hidden, cell)= self.encoder_model(embeddings_varying_encoder, (input_hidden, input_cell), lengths= encoder_lengths, enforce_sorted= False)
            encoder_output= self.BiLSTM_dense_layer_encoder(encoder_output)
            # run local decoder
            decoder_output, _= self.decoder_model(embeddings_varying_decoder, (hidden, cell), lengths= decoder_lengths, enforce_sorted= False)
            decoder_output= self.BiLSTM_dense_layer_decoder(decoder_output)
        elif self.encoder_and_decoder_model== "GRU":
            # GRU
            # calculate initial state
            input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)
            # run local encoder
            encoder_output, hidden= self.encoder_model(embeddings_varying_encoder, input_hidden)
            # run local decoder
            decoder_output, _= self.decoder_model(embeddings_varying_decoder, hidden)
        elif self.encoder_and_decoder_model== "BiGRU":
            # BiGRU
            # calculate initial state
            input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(self.hparams.lstm_layers* 2, -1, -1)
            encoder_output, hidden= self.encoder_model(embeddings_varying_encoder, input_hidden)
            encoder_output= self.GRU_dense_layer_encoder(encoder_output)
            # run local decoder
            decoder_output, _= self.decoder_model(embeddings_varying_decoder, hidden)
            decoder_output= self.GRU_dense_layer_decoder(decoder_output)
        else:
            raise ValueError("model: {} is not provide in this versioin".format(self.encoder_and_decoder_model))
            
        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0:
            self.log_interpretation(outputs)

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # take attention and concatenate if a list to proper attention object
        if isinstance(out["decoder_attention"], (list, tuple)):
            batch_size = len(out["decoder_attention"])
            # start with decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
            first_elm = out["decoder_attention"][0]
            # create new attention tensor into which we will scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["decoder_attention"]):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]

            # same game for encoder attention
            # create new attention tensor into which we will scatter
            first_elm = out["encoder_attention"][0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.hparams.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            for idx, x in enumerate(out["encoder_attention"]):
                encoder_length = out["encoder_lengths"][idx]
                encoder_attention[idx, :, :, self.hparams.max_encoder_length - encoder_length :] = x[
                    ..., :encoder_length
                ]
        else:
            decoder_attention = out["decoder_attention"]
            decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
            decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")
            # roll encoder attention (so start last encoder value is on the right)
            encoder_attention = out["encoder_attention"]
            shifts = encoder_attention.size(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                    encoder_attention
                )
                - shifts[:, None, None, None]
            ) % encoder_attention.size(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attentiont to full size
            if encoder_attention.size(-1) < self.hparams.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.hparams.max_encoder_length - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = masked_op(
            attention[
                :, attention_prediction_horizon, :, : self.hparams.max_encoder_length + attention_prediction_horizon
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            attention = masked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / masked_op(attention, dim=1, op="sum").unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.interpret_output(out.iget(slice(idx, idx + 1)))
            for f in to_list(fig):
                ax = f.axes[0]
                ax2 = ax.twinx()
                ax2.set_ylabel("Attention")
                encoder_length = x["encoder_lengths"][0]
                ax2.plot(
                    torch.arange(-encoder_length, 0),
                    interpretation["attention"][0, -encoder_length:].detach().cpu(),
                    alpha=0.2,
                    color="k",
                )
                f.tight_layout()
        return fig

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), self.decoder_variables
        )

        return figs

    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack([x["interpretation"][name].detach() for x in outputs], side="right", value=0).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).cumsum(0).float()
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = self.current_stage
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs])
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )

    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=self.global_step
            )