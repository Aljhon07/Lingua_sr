import matplotlib.pyplot as plt
import numpy as np

def plot_original_vs_normalized(self, original_spectrogram, normalized_spectrogram):
        """Plots the original and normalized spectrograms side-by-side."""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Original Spectrogram
        im1 = axes[0].imshow(original_spectrogram.squeeze(0), cmap='inferno', origin='lower', aspect='auto', interpolation='none')
        axes[0].set_title("Original Mel Spectrogram")
        axes[0].set_ylabel("Frequency (Mel Bands)")
        axes[0].set_xlabel("Time (Frames)")
        fig.colorbar(im1, ax=axes[0])  # Use default colorbar for original

        # Normalized Spectrogram
        im2 = axes[1].imshow(normalized_spectrogram.squeeze(0), cmap='inferno', origin='lower', aspect='auto', interpolation='none')
        axes[1].set_title("Normalized Mel Spectrogram")
        axes[1].set_ylabel("Frequency (Mel Bands)")
        axes[1].set_xlabel("Time (Frames)")
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()