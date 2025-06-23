import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import optuna
from optuna.samplers import TPESampler
import pickle
import os
from datetime import datetime

logger = logging.getLogger()
logger.addHandler(logging.FileHandler("optuna.log", mode="w"))
optuna.logging.enable_propagation()  # Propagate logs to the root logger.

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesGenerator(nn.Module):
    """Generator network for time series data"""
    def __init__(self, noise_dim=100, hidden_dim=256, output_dim=1440, num_layers=3):
        super(TimeSeriesGenerator, self).__init__()

        layers = []
        input_dim = noise_dim

        # Build the generator layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

class TimeSeriesDiscriminator(nn.Module):
    """Discriminator network for time series data"""
    def __init__(self, input_dim=1440, hidden_dim=256, num_layers=3):
        super(TimeSeriesDiscriminator, self).__init__()

        layers = []
        current_dim = input_dim

        # Build discriminator layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class TimeSeriesGAN:
    """Complete GAN implementation for time series data"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize networks
        self.generator = TimeSeriesGenerator(
            noise_dim=config['noise_dim'],
            hidden_dim=config['gen_hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['gen_layers']
        ).to(self.device)

        self.discriminator = TimeSeriesDiscriminator(
            input_dim=config['output_dim'],
            hidden_dim=config['disc_hidden_dim'],
            num_layers=config['disc_layers']
        ).to(self.device)

        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['gen_lr'],
            betas=(0.5, 0.999)
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['disc_lr'],
            betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'val_g_loss': [],
            'val_d_loss': []
        }

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config['min_delta']
        )

    def load_data(self, csv_path, test_size=0.2):
        """Load and preprocess time series data from CSV"""
        print(f"Loading data from {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path, header=None, skiprows=1)
        print(f"Data shape: {df.shape}")
        data = df.values.astype(np.float32)

        print(f"Data shape: {data.shape}")

        # Normalize data to [-1, 1] range
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = self.scaler.fit_transform(data)

        # Split data
        train_data, val_data = train_test_split(
            data_normalized,
            test_size=test_size,
            random_state=42
        )

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(train_data))
        val_dataset = TensorDataset(torch.FloatTensor(val_data))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        g_losses = []
        d_losses = []

        for batch_idx, (real_data,) in enumerate(self.train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(self.device)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()

            # Real data
            d_real = self.discriminator(real_data)
            d_real_loss = self.criterion(d_real, real_labels)

            # Fake data
            noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
            fake_data = self.generator(noise)
            d_fake = self.discriminator(fake_data.detach())
            d_fake_loss = self.criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()

            # Generate fake data and try to fool discriminator
            noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
            fake_data = self.generator(noise)
            d_fake = self.discriminator(fake_data)
            g_loss = self.criterion(d_fake, real_labels)  # Want discriminator to think it's real

            g_loss.backward()
            self.g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        return np.mean(g_losses), np.mean(d_losses)

    def validate(self):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()

        g_losses = []
        d_losses = []

        with torch.no_grad():
            for real_data, in self.val_loader:
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)

                # Create labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Discriminator validation
                d_real = self.discriminator(real_data)
                d_real_loss = self.criterion(d_real, real_labels)

                noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
                fake_data = self.generator(noise)
                d_fake = self.discriminator(fake_data)
                d_fake_loss = self.criterion(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                # Generator validation
                g_loss = self.criterion(d_fake, real_labels)

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

        return np.mean(g_losses), np.mean(d_losses)

    def save_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'scaler': self.scaler
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f'gan_checkpoint_epoch_{epoch}_{timestamp}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.history = checkpoint['history']
        self.scaler = checkpoint['scaler']

        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']

    def train(self, epochs, checkpoint_freq=10):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(1, epochs + 1):
            # Training
            g_loss, d_loss = self.train_epoch()

            # Validation
            val_g_loss, val_d_loss = self.validate()

            # Update history
            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            self.history['val_g_loss'].append(val_g_loss)
            self.history['val_d_loss'].append(val_d_loss)

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
                print(f"  Val   - G Loss: {val_g_loss:.4f}, D Loss: {val_d_loss:.4f}")

            # Save checkpoint
            if epoch % checkpoint_freq == 0:
#                self.save_checkpoint(epoch)
                pass

            # Early stopping
            if False and self.early_stopping(val_g_loss, self.generator):
                print(f"Early stopping at epoch {epoch}")
                break

        print("Training completed!")

    def generate_samples(self, num_samples=100):
        """Generate synthetic time series samples"""
        self.generator.eval()

        with torch.no_grad():
            noise = torch.randn(num_samples, self.config['noise_dim']).to(self.device)
            fake_data = self.generator(noise)

            # Denormalize data
            fake_data_np = fake_data.cpu().numpy()
            fake_data_denorm = self.scaler.inverse_transform(fake_data_np)

        return fake_data_denorm

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['g_loss'], label='Generator Train')
        plt.plot(self.history['val_g_loss'], label='Generator Val')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['d_loss'], label='Discriminator Train')
        plt.plot(self.history['val_d_loss'], label='Discriminator Val')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

def objective(trial):
    """Objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    config = {
        'epochs': trial.suggest_int('epochs', 10, 100, step=5),
        'noise_dim': trial.suggest_int('noise_dim', 50, 200),
        'gen_hidden_dim': trial.suggest_int('gen_hidden_dim', 128, 512, step=64),
        'disc_hidden_dim': trial.suggest_int('disc_hidden_dim', 128, 512, step=64),
        'gen_layers': trial.suggest_int('gen_layers', 2, 5),
        'disc_layers': trial.suggest_int('disc_layers', 2, 5),
        'gen_lr': trial.suggest_float('gen_lr', 1e-5, 1e-2, log=True),
        'disc_lr': trial.suggest_float('disc_lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'output_dim': 1440,
        'patience': 15,
        'min_delta': 0.001
    }

    # Train model with suggested hyperparameters
    gan = TimeSeriesGAN(config)
    gan.load_data('data/household_power_consumption-valid.txt')  # Replace with your CSV file path

    # Train for fewer epochs during hyperparameter search
    gan.train(config['epochs'], checkpoint_freq=50)

    # Return validation loss as objective (minimize)
    return gan.history['val_g_loss'][-1] + gan.history['val_d_loss'][-1] + gan.history['g_loss'][-1] + gan.history['d_loss'][-1]

def hyperparameter_optimization(n_trials=20):
    """Run hyperparameter optimization"""
    print("Starting hyperparameter optimization...")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params

def main():
    """Main function to run the complete pipeline"""

    # Option 1: Use default configuration
    default_config = {
        'epochs': 50,
        'noise_dim': 183,
        'gen_hidden_dim': 128,
        'disc_hidden_dim': 384,
        'gen_layers': 4,
        'disc_layers': 5,
        'gen_lr': 1.3421950061292506e-05,
        'disc_lr': 1.9289435865650024e-05,
        'batch_size': 128,
        'output_dim': 1440,
        'patience': 20,
        'min_delta': 0.001
    }

    # Option 2: Run hyperparameter optimization (uncomment to use)
    if False:
        print("Running hyperparameter optimization...")
        best_config = hyperparameter_optimization(n_trials=2000)
        best_config['output_dim'] = 1440
        best_config['patience'] = 20
        best_config['min_delta'] = 0.001
        config = best_config
    else:
        config = default_config

    # Initialize and train GAN
    gan = TimeSeriesGAN(config)

    # Load your data (replace 'your_data.csv' with your actual file path)
    gan.load_data('data/household_power_consumption-valid.txt')

    # Train the model
    gan.train(config['epochs'], checkpoint_freq=20)

    # Plot training history
    gan.plot_training_history()

    # Generate synthetic samples
    print("Generating synthetic samples...")
    synthetic_data = gan.generate_samples(num_samples=20*365+5)
    print(f"Generated {synthetic_data.shape[0]} synthetic samples")

    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv('data\generated_data_gan.csv', index=False)
    print("Synthetic data saved to 'data\generated_data_gan.csv'")

if __name__ == "__main__":
    main()