from data_parser import ParsedFile
from matplotlib import pyplot as plt

SAMPLE_FREQUENCY = 4000 # 0.25 ms per sample

class WindowedData:
    '''Class to window the data from a ParsedFile object.'''

    def __init__(self, ParsedFile, window_size, overlap):
        '''
        Initialize the WindowedData object.
        Parameters:
            ParsedFile: The ParsedFile object containing the data to be windowed.
            window_size: The size of the window in ms.
            overlap: The overlap between windows in ms.
        '''
        self.window_size = window_size
        self.overlap = overlap
        self.parsed_file = ParsedFile

        self.windowed_inputs = []
        self.windowed_classes = []

        # Define how many samples will be in each window (window size in ms)
        window_samples = window_size * SAMPLE_FREQUENCY // 1000

        # Define how many samples will overlap between windows (overlap in ms)
        overlap_samples = overlap * SAMPLE_FREQUENCY // 1000

        # Create windows
        for i in range(0, len(self.parsed_file.inputs_c1) - window_samples, overlap_samples):
            # One window will consists of the inputs from channel 1 followed by the inputs from channel 2
            self.windowed_inputs.append(self.parsed_file.inputs_c1[i:i+window_samples] + self.parsed_file.inputs_c2[i:i+window_samples])
            self.windowed_classes.append(sum(self.parsed_file.classes[i:i+window_samples]) / window_samples)

    def getWindows(self):
        '''Return the windowed inputs and classes.'''
        return self.windowed_inputs, self.windowed_classes
    
    def draw(self):
        '''Draw the windowed data. (For debugging purposes. It just shows the time overlap not the actual data.)'''
        y_offset = 1
        window_samples = self.window_size * SAMPLE_FREQUENCY // 1000
        overlap_samples = self.overlap * SAMPLE_FREQUENCY // 1000

        plt.figure(figsize=(10, 15))

        for i in range(0, len(self.parsed_file.inputs) - window_samples, overlap_samples):
            segment = [i, i + window_samples]
            y = [1 + y_offset, 1 + y_offset]
            plt.plot(segment, y)
            y_offset += 1
        
        plt.title('Windowed Data')
        plt.savefig(f'plots/windows.png')


# Test
#data = ParsedFile('/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/2024_4_6_TestSub20_ARM_L_117.csv')
#windowed_data = WindowedData(data, 250, 190)
#windowed_data.draw()
