# Contains training helper functions like early stopping, calculation of saliency maps, etc.
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True


def calculate_average_saliency_maps(model, lab_frame, ronen_frame):
    num_outputs = model(lab_frame, ronen_frame).size(1)
    accumulated_lab_saliency = 0
    accumulated_ronen_saliency = 0

    saliency = Saliency(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for output_idx in range(num_outputs):
            lab_saliency, ronen_saliency = saliency.attribute((lab_frame, ronen_frame), target=output_idx)
            accumulated_lab_saliency += lab_saliency
            accumulated_ronen_saliency += ronen_saliency

    average_lab_saliency = accumulated_lab_saliency / num_outputs
    average_ronen_saliency = accumulated_ronen_saliency / num_outputs

    return average_lab_saliency.squeeze().cpu().detach().numpy(), average_ronen_saliency.squeeze().cpu().detach().numpy()


def plot_saliency_maps(lab_frame_normalized, ronen_frame_normalized, lab_saliency, ronen_saliency):
    batch_size = lab_saliency.shape[0]
    plt.figure(figsize=(12, 2 * batch_size))

    for i in range(batch_size // 2):
        # Display Lab Frame
        plt.subplot(batch_size, 4, 4 * i + 1)
        plt.title(f"Lab Frame {i + 1}")
        plt.imshow(lab_frame_normalized[i].permute(1, 2, 0))
        plt.axis('off')

        # Display Lab Saliency Map
        plt.subplot(batch_size, 4, 4 * i + 2)
        plt.title(f"Lab Saliency Map {i + 1}")
        plt.imshow(np.transpose(lab_saliency[i], (1, 2, 0)), cmap='viridis')
        plt.axis('off')

        # Display Ronen Frame
        plt.subplot(batch_size, 4, 4 * i + 3)
        plt.title(f"Ronen Frame {i + 1}")
        plt.imshow(ronen_frame_normalized[i].permute(1, 2, 0))
        plt.axis('off')

        # Display Ronen Saliency Map
        plt.subplot(batch_size, 4, 4 * i + 4)
        plt.title(f"Ronen Saliency Map {i + 1}")
        plt.imshow(np.transpose(ronen_saliency[i], (1, 2, 0)), cmap='viridis')
        plt.axis('off')

    plt.tight_layout()
    plt.show()