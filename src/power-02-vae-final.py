import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    def __init__(self, data, scaler=None, fit_scaler=True):
        if isinstance(data, pd.DataFrame):
            data = data.values

        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

        if fit_scaler:
            self.data = self.scaler.fit_transform(data)
        else:
            self.data = self.scaler.transform(data)

        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VAE(nn.Module):
    """Variational Autoencoder for time series data"""
    def __init__(self, input_dim=1440, hidden_dims=[512, 256], latent_dim=64, dropout=0.2):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence and reconstruction loss"""
    # Reconstruction loss (MSE for continuous data)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class VAETrainer:
    """Trainer class for VAE with checkpointing capabilities"""
    def __init__(self, model, train_loader, val_loader, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0

    def save_checkpoint(self, epoch, optimizer, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {loss:.4f}", False)

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epochs_trained = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.epochs_trained}")
        return checkpoint['loss']

    def train_epoch(self, optimizer, beta=1.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = self.model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(self.train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(self.train_loader.dataset)

        return avg_loss, avg_recon_loss, avg_kl_loss

    def validate(self, beta=1.0):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_recon_loss = total_recon_loss / len(self.val_loader.dataset)
        avg_kl_loss = total_kl_loss / len(self.val_loader.dataset)

        return avg_loss, avg_recon_loss, avg_kl_loss

    def train(self, epochs, lr=1e-3, beta=1.0, save_every=10, patience=20):
        """Train the VAE"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        epochs_without_improvement = 0

        for epoch in range(self.epochs_trained + 1, self.epochs_trained + epochs + 1):
            # Training
            train_loss, train_recon, train_kl = self.train_epoch(optimizer, beta)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_recon, val_kl = self.validate(beta)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Print progress
            if epoch % 5 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                      f'Train Recon: {train_recon:.4f} | Train KL: {train_kl:.4f}')

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
#                self.save_checkpoint(epoch, optimizer, val_loss, is_best)
                pass

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch} epochs')
                break

        self.epochs_trained = epoch
        return self.best_val_loss

def objective(trial, train_loader, val_loader, device, input_dim=1440):
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 16, 128, step=16)
    hidden_dim1 = trial.suggest_int('hidden_dim1', 256, 1024, step=128)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 128, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    beta = trial.suggest_float('beta', 0.1, 2.0)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Create model
    model = VAE(
        input_dim=input_dim,
        hidden_dims=[hidden_dim1, hidden_dim2],
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    # Create trainer
    trainer = VAETrainer(model, train_loader, val_loader, device)

    # Train model
    try:
        best_loss = trainer.train(epochs=50, lr=lr, beta=beta, save_every=50, patience=10)
        return best_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def load_data(csv_path, test_size=0.2, val_size=0.1):
    """Load and preprocess time series data from CSV"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    print(f"Data shape: {df.shape}")

    # Split data
    train_data, temp_data = train_test_split(df, test_size=test_size + val_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + val_size), random_state=42)

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, fit_scaler=True)
    val_dataset = TimeSeriesDataset(val_data, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = TimeSeriesDataset(test_data, scaler=train_dataset.scaler, fit_scaler=False)

    return train_dataset, val_dataset, test_dataset

def generate_synthetic_data(model, scaler, n_samples=100, device='cpu'):
    """Generate synthetic time series data using trained VAE"""
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(n_samples, model.latent_dim).to(device)

        # Decode to generate data
        generated = model.decode(z)

        # Inverse transform to original scale
        generated_np = generated.cpu().numpy()
        generated_original = scaler.inverse_transform(generated_np)

    return generated_original

def plot_training_history(trainer, save_path='training_history.png'):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')

    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_losses[-50:], label='Training Loss (Last 50)')
    plt.plot(trainer.val_losses[-50:], label='Validation Loss (Last 50)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Recent Training History')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Configuration
    CSV_PATH = 'data/household_power_consumption-valid.txt'  # Replace with your CSV file path
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    EPOCHS = 100

    print(f"Using device: {DEVICE}")

    # Load data
    try:
        train_dataset, val_dataset, test_dataset = load_data(CSV_PATH)
    except FileNotFoundError:
        print(f"CSV file not found: {CSV_PATH}")
        print("Please ensure your CSV file exists and update the CSV_PATH variable")
        return

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Hyperparameter optimization (optional)
    if False:
        print("\nStarting hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, DEVICE),
            n_trials=2000,
            timeout=10*3600  # 1 hour timeout
        )

        print("Best hyperparameters:")
        print(study.best_params)

        # Train final model with best hyperparameters
        best_params = study.best_params
        model = VAE(
            input_dim=1440,
            hidden_dims=[best_params['hidden_dim1'], best_params['hidden_dim2']],
            latent_dim=best_params['latent_dim'],
            dropout=best_params['dropout']
        ).to(DEVICE)
    else:
        # Train final model with fixed hyperparameters
        model = VAE(
            input_dim=1440,
            hidden_dims=[1024, 512],
            latent_dim=64,
            dropout=0.10744608291873486
        ).to(DEVICE)

    print(f"\nTraining final model...")
    trainer = VAETrainer(model, train_loader, val_loader, DEVICE)

    # Train the model
    best_loss = trainer.train(
        epochs=EPOCHS,
        lr=0.00022639934529815903,
        beta=0.10066011665291091,
        save_every=20,
        patience=30
    )

    print(f"\nTraining completed. Best validation loss: {best_loss:.4f}")

    # Load best model
    best_model_path = os.path.join(trainer.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
        print("Loaded best model for inference")

    # Plot training history
    plot_training_history(trainer)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_data = generate_synthetic_data(
        model, train_dataset.scaler, n_samples=20*365+5, device=DEVICE
    )

    print(f"Generated {synthetic_data.shape[0]} synthetic time series")
    print(f"Synthetic data shape: {synthetic_data.shape}")

    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv('data/generated_data_vae.csv', index=False)
    print("Synthetic data saved to 'data/generated_data_vae.csv'")

    # Evaluate model on test set
    test_loss, test_recon, test_kl = trainer.validate()
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Reconstruction Loss: {test_recon:.4f}")
    print(f"Test KL Loss: {test_kl:.4f}")

if __name__ == "__main__":
    main()