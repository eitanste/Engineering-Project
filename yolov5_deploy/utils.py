import cv2
import matplotlib.pyplot as plt
import numpy as np


def list_active_video_inputs(max_devices=10):
    active_devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            device_info = {
                'index': i,
                'frame_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'frame_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                'backend_name': cap.getBackendName(),
            }
            active_devices.append(device_info)
            cap.release()
    return active_devices


def get_input_streams():
    # global device
    active_video_inputs = list_active_video_inputs()
    for device in active_video_inputs:
        print(f"Device Index: {device['index']}")
        print(f"  Frame Width: {device['frame_width']}")
        print(f"  Frame Height: {device['frame_height']}")
        print(f"  FPS: {device['fps']}")
        fourcc = int(device['fourcc'])
        print(f"  FourCC: {fourcc & 0xFF} {fourcc >> 8 & 0xFF} {fourcc >> 16 & 0xFF} {fourcc >> 24 & 0xFF}")
        print(f"  Backend Name: {device['backend_name']}")
        print()
    # Example output processing
    if not active_video_inputs:
        print("No active video inputs found.")
    else:
        print(f"Found {len(active_video_inputs)} active video inputs.")


def frame_image_with_header(image_path, header_text):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image '{image_path}'. Check the file path.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = img.shape

    # Create a white border
    border_size = 0  # Narrow border size
    img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                         value=[255, 255, 255])

    # Plot the image with a header
    plt.figure(figsize=(width / 100, (height) / 100))  # Add extra space for the header
    plt.imshow(img_with_border)
    plt.title(header_text, fontsize=16, pad=20)
    plt.axis('off')  # Remove the axis

    # Save the plot as an SVG file
    output_file_svg = 'framed_image_with_header.svg'
    plt.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print(f'Saved framed image with header as {output_file_svg}')

    # Save the plot as a PNG file
    output_file_png = 'framed_image_with_header.png'
    plt.savefig(output_file_png, format='png', dpi=300, bbox_inches='tight')
    print(f'Saved framed image with header as {output_file_png}')

    plt.show()


def plot_bar_chart(categories, labels, accuracies, confidences, figsize=(5.19, 4.51), bar_width=0.6,
                   output_prefix='label_accuracy_confidence'):
    """
    Plots a bar chart with given data and saves it as PNG and SVG files.

    Args:
        categories (list): List of category names for the legend.
        labels (list): List of labels for the x-axis.
        accuracies (list): List of accuracy values for the bars.
        confidences (list): List of confidence values for the bars.
        figsize (tuple): Size of the figure (width, height) in inches.
        bar_width (float): Width of the bars.
        output_prefix (str): Prefix for the output file names.
    """
    # Prepare the data for plotting
    x = np.arange(len(labels))
    colors = ['blue'] * 3 + ['green'] * 3 + ['red'] * 2 + ['purple']

    # Create the bar plot with the specified aspect ratio
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, accuracies, bar_width, color=colors)

    # Add text annotations for confidence levels
    for bar, confidence in zip(bars, confidences):
        height = bar.get_height()
        plt.annotate(f'confidence = {confidence:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                     xytext=(0, 0),  # Center the text
                     textcoords="offset points",
                     ha='center', va='center', fontsize=16, color='white', rotation=90)

    # Add text annotations for accuracies on top of the bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.annotate(f'{accuracy:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=14, color='black')

    # Add labels, title, and custom x-axis labels
    ax.set_xlabel('Label', fontsize=22)
    ax.xaxis.set_label_coords(0.125, -0.14)  # Move the x-axis label to the left
    ax.set_ylabel('Accuracy (%)', fontsize=22)
    ax.set_title('Accuracy and Confidence Level per Labels (Medium model)', fontsize=24)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=12)

    # Add legend and grid
    # Create custom legend
    custom_lines = [plt.Line2D([0], [0], color='blue', lw=4),
                    plt.Line2D([0], [0], color='green', lw=4),
                    plt.Line2D([0], [0], color='red', lw=4),
                    plt.Line2D([0], [0], color='purple', lw=4)]
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(custom_lines, categories, fontsize=16, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1),
               frameon=False)

    # Save the plot in high quality
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', format='png', dpi=300)
    plt.savefig(f'{output_prefix}.svg', format='svg')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # get_input_streams()

    ## frame_image_with_header
    # image_path = r'C:\Users\eitan\OneDrive\Desktop\Master\study\B.A\Year_4\final_project\Engineering-Project\yolov5_deploy\pose_estimation.png'  # Replace with the path to your image
    # header_text = 'YOLOv5 - Pose Estimation'
    # frame_image_with_header(image_path, header_text)

    # plot_bar_chart
    categories = ['Choking Objects', 'Sharp Objects', 'Hot Objects', 'Others']
    labels = ['bottle', 'carrot', 'broccoli', 'Vase', 'wine glass', 'knife', 'cup', 'oven', 'human']
    accuracies = [85, 82, 94, 86, 92, 70, 84, 86, 98]
    confidences = [82, 75, 84, 81, 86, 67, 89, 90, 95]

    plot_bar_chart(categories, labels, accuracies, confidences)
