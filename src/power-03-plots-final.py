import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path):
    """Load and preprocess data from CSV file"""
    # Load CSV
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    print(f"Loaded data shape: {df.shape}")

    # Convert to numpy array
    data = df.values.astype(np.float32)

    return data

def plot_data(filename, datax, datay):
    # Create the plot
    plt.figure(figsize=(19, 10))
    plt.plot(datax, datay, label='Power Intensity', alpha=0.7)
    plt.ylim(0, 12)
    plt.title('Power over Time')
    plt.xlabel('MinuteOfDay')
    plt.ylabel('Power')
    plt.legend()
    plt.grid()

    # Save the plot to an image file
    plt.savefig(filename)
    plt.close()
    print (f"Generated plot: {filename}")

# Numbers from 0 to 1439 for the X Axis
time_array = np.arange(1440)

# Generate some plots with the original data
data_orig = load_data("data/household_power_consumption-valid.txt")
plot_data("data/original_day1.png", time_array, data_orig[0])
plot_data("data/original_day101.png", time_array, data_orig[100])
plot_data("data/original_day201.png", time_array, data_orig[200])
plot_data("data/original_day301.png", time_array, data_orig[300])
plot_data("data/original_day401.png", time_array, data_orig[400])
plot_data("data/original_day501.png", time_array, data_orig[500])
plot_data("data/original_day757.png", time_array, data_orig[756])
plot_data("data/original_day758.png", time_array, data_orig[757])

# Generate some plots with the generated data from the GAN
data_gan = load_data("data/generated_data_gan.csv")
plot_data("data/generated_gan_1.png", time_array, data_gan[0])
plot_data("data/generated_gan_2.png", time_array, data_gan[1])
plot_data("data/generated_gan_3.png", time_array, data_gan[2])
plot_data("data/generated_gan_4.png", time_array, data_gan[3])
plot_data("data/generated_gan_5.png", time_array, data_gan[4])
plot_data("data/generated_gan_6.png", time_array, data_gan[5])

# Generate some plots with the generated data from the VAE
data_vae = load_data("data/generated_data_vae.csv")
plot_data("data/generated_vae_1.png", time_array, data_vae[0])
plot_data("data/generated_vae_2.png", time_array, data_vae[1])
plot_data("data/generated_vae_3.png", time_array, data_vae[2])
plot_data("data/generated_vae_4.png", time_array, data_vae[3])
plot_data("data/generated_vae_5.png", time_array, data_vae[4])
plot_data("data/generated_vae_6.png", time_array, data_vae[5])
