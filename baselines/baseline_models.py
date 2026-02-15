"""
Baseline Models for Weather Forecasting
Used as benchmark for NAS-discovered architectures
"""

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM baseline model
    
    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        output_size: Number of output values (forecast horizon)
        dropout: Dropout rate (default: 0.2)
        bidirectional: Use bidirectional LSTM (default: True)
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 output_size=1, dropout=0.2, bidirectional=True):
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
        
        Returns:
            predictions: Output tensor (batch_size, output_size)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            last_hidden = hidden[-1,:,:]
        
        # Dropout and fully connected
        out = self.dropout(last_hidden)
        predictions = self.fc(out)
        
        return predictions
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineGRU(nn.Module):
    """
    2-layer GRU baseline model
    Similar architecture to LSTM but using GRU cells
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2):
        super(BaselineGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1,:,:]
        out = self.dropout(last_hidden)
        predictions = self.fc(out)
        return predictions
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineAttention(nn.Module):
    """
    Transformer-based baseline with multi-head attention
    """
    
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2,
                 output_size=1, dropout=0.2):
        super(BaselineAttention, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output
        x = self.dropout(x)
        predictions = self.fc(x)
        
        return predictions
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model
    CNN for feature extraction, LSTM for temporal modeling
    """
    
    def __init__(self, input_size, cnn_filters=64, lstm_hidden=128,
                 output_size=1, dropout=0.2):
        super(HybridCNNLSTM, self).__init__()
        
        self.input_size = input_size
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Output
        self.fc = nn.Linear(lstm_hidden, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # CNN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1,:,:]
        
        # Output
        out = self.dropout(last_hidden)
        predictions = self.fc(out)
        
        return predictions
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleMLP(nn.Module):
    """
    Simple 3-layer MLP baseline (for efficiency comparison)
    """
    
    def __init__(self, input_size, seq_length, hidden_sizes=[256, 128],
                 output_size=1, dropout=0.2):
        super(SimpleMLP, self).__init__()
        
        # Flatten input
        flat_input = input_size * seq_length
        
        layers = []
        prev_size = flat_input
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_baseline_model(model_name, input_size, seq_length=24, output_size=1):
    """
    Factory function to get baseline model by name
    
    Args:
        model_name: Name of model ('lstm', 'gru', 'attention', 'hybrid', 'mlp')
        input_size: Number of input features
        seq_length: Sequence length
        output_size: Number of outputs
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'lstm':
        return BaselineLSTM(input_size, output_size=output_size)
    elif model_name == 'gru':
        return BaselineGRU(input_size, output_size=output_size)
    elif model_name == 'attention':
        return BaselineAttention(input_size, output_size=output_size)
    elif model_name == 'hybrid':
        return HybridCNNLSTM(input_size, output_size=output_size)
    elif model_name == 'mlp':
        return SimpleMLP(input_size, seq_length, output_size=output_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
