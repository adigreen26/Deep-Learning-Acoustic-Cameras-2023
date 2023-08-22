# Contains helper functions like plot label distribution, display random frame, etc.
import random
import matplotlib.pyplot as plt


def plot_labels(train_labels, test_labels):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the training set labels in blue
    ax.scatter(train_labels['x'], train_labels['y'], train_labels['z'], color='b', alpha=0.5, label='Train')

    # Plot the test set labels in red
    ax.scatter(test_labels['x'], test_labels['y'], test_labels['z'], color='r', alpha=0.5, label='Test')

    # Set the labels of the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title of the plot
    ax.set_title('Label Distribution')

    # Add a legend
    ax.legend()

    plt.show()

def display_random_frame(dataset):
    # Get a random index
    idx = random.randint(0, len(dataset) - 1)

    # Get the videos and label at this index
    lab_video, ronen_video, label = dataset[idx]

    # Get the first frame of each video
    lab_frame = lab_video  # Assuming the first frame is at index 0
    ronen_frame = ronen_video  # Assuming the first frame is at index 0

    # Normalize the frames to [0, 1]
    lab_frame_normalized = lab_frame / 255.0
    ronen_frame_normalized = ronen_frame / 255.0

    # Display the frames
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(lab_frame_normalized)
    axs[0].set_title('Lab Frame')

    axs[1].imshow(ronen_frame_normalized)
    axs[1].set_title('Ronen Frame')

    plt.show()