import csv
from os import makedirs
from os.path import basename
from matplotlib import pyplot as plt
import numpy as np

class ParsedFile:
    def __init__(self, filename):
        self.filename = filename
        self.metadata, self.inputs, self.classes = parse_datafile(filename)
        self.inputs = [[float(x) for x in input] for input in self.inputs]
        self.classes = [int(cls) for cls in self.classes]

    def draw(self):
        makedirs('./plots', exist_ok=True)

        # Plot inputs
        inputs = list(zip(*self.inputs))
        num_points = len(inputs[0])
        chunk_size = num_points // 8

        for i in range(8):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            fig, ax1 = plt.subplots(figsize=(20, 10))

            ax1.set_xlabel('Channel 1 Data Point')
            ax1.set_ylabel('Amplitude')
            ax1.plot(range(start, end), inputs[0][start:end], color='blue', label='Channel 1', linewidth=0.5)
            ax1.plot(range(start, end), inputs[1][start:end], color='red', label='Channel 2', linewidth=0.5)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Class')
            ax2.plot(range(start, end), self.classes[start:end], color='green', label='Class', linewidth=1)

            ax1.legend()
            ax2.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(f'plots/{basename(self.filename)}_{i+1}.png')


def parse_datafile(filename):
    metadata = {}
    inputs = []
    classes = []
    with open(filename) as datafile:
        reader = csv.reader(datafile, delimiter=';')

        # Read metadata
        for _ in range(4):
            line = next(reader)
            key, value = line[0].split(':')
            metadata[key.strip()] = value.strip()

        # Skip data type line
        next(reader)

        # Read data
        for row in reader:
            inputs.append([row[1], row[2]])
            classes.append(row[-1])

    return metadata, inputs, classes

# Test
#data1 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_119.csv')
#data1.draw()

#data2 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_118.csv')
#data2.draw()

#data3 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_117.csv')
#data3.draw()