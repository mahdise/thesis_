import matplotlib.pyplot as plt
import numpy as np
from Evaluation.utils import read_data

def plot_for_column(data_path, column_name, gt_value, gt_name):

    data_frame = read_data(data_path)

    print(data_frame)
    height = list(data_frame[column_name])
    height.pop(0)
    height.append(gt_value)
    bars = list(data_frame["Unnamed: 0"])
    bars.pop(0)
    bars.append(gt_name)

    # create a dataset

    x_pos = np.arange(len(bars))

    # Create bars with different colors
    plt.bar(x_pos, height, color=['maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'blue'])

    # Create names on the x-axis
    plt.xticks(x_pos, bars)
    plt.xlabel("Statistical analysis name")
    plt.ylabel("Value")
    plt.title("Statistical analysis with ground truth")

    # Show graph
    # plt.show()
    plt.savefig('/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/Plot/1111/st_analysis.png')


if __name__ == '__main__':
    plot_for_column("/Evaluation/StatisticalData/statistical_preprocessed_orientation_result_1111.csv",
                    column_name="rotate_angle",
                    gt_value=119,
                    gt_name="gt")