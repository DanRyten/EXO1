import csv
from os import makedirs
from os.path import basename
from matplotlib import pyplot as plt

DATA_SPLITS = 8 # Number of splits to divide the data for plotting

class ParsedFile:
    '''
    Class to parse the data from the EMG csv files.
    '''

    def __init__(self, filename):
        '''
        Initialize the ParsedFile object.
        Parameters:
            filename: The path of the csv file to parse.
        '''
        self.filename = filename
        self.inputs_c1, self.inputs_c2, self.classes = parse_datafile(filename)
        self.inputs_c1 = [float(input) for input in self.inputs_c1]
        self.inputs_c2 = [float(input) for input in self.inputs_c2]
        self.classes = [int(cls) for cls in self.classes]

    def draw(self):
        '''
        Draw the data. (Mostly for data analysis purposes. It shows both channels and the corresponding class.)
        '''
        makedirs('./plots', exist_ok=True)

        # Plot inputs
        num_points = len(self.inputs_c1)
        chunk_size = num_points // DATA_SPLITS

        for i in range(DATA_SPLITS):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            fig, ax1 = plt.subplots(figsize=(20, 10))

            ax1.set_xlabel('Channel 1 Data Point')
            ax1.set_ylabel('Amplitude')
            ax1.plot(range(start, end), self.inputs_c1[start:end], color='blue', label='Channel 1', linewidth=0.5)
            ax1.plot(range(start, end), self.inputs_c2[start:end], color='red', label='Channel 2', linewidth=0.5)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Class')
            ax2.plot(range(start, end), self.classes[start:end], color='green', label='Class', linewidth=1)

            ax1.legend()
            ax2.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(f'plots/{basename(self.filename)}_{i+1}.png')


def parse_datafile(filename):
    '''
    Parse the data from the csv file.
    Parameters:
        filename: The path of the csv file to parse.
    Returns:
        inputs_c1: A list containing the inputs from channel 1.
        inputs_c2: A list containing the inputs from channel 2.
        classes: A list containing the classes.
    '''
    inputs_c1 = []
    inputs_c2 = []
    classes = []
    with open(filename) as datafile:
        reader = csv.reader(datafile, delimiter=';')

        # Read data
        for row in reader:
            inputs_c1.append(row[1])
            inputs_c2.append(row[2])
            classes.append(row[-1])

    return inputs_c1, inputs_c2, classes

# Test
#data1 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_119.csv')
#data1.draw()

#data2 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_118.csv')
#data2.draw()

#data3 = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_117.csv')
#data3.draw()