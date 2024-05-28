# How to visualize the different data files

There are some commented lines at the bottom of the data_parser.py file with examples.

Steps:
1. Load the data and store it in a ParsedFile object
    - loadedData = ParsedFile('pathToFile')
2. Call the draw method from the created ParsedFile object
    - loadedData.draw()

The created plots will be stored in a folder called "plots" in the current working folder.

# Split the data for better resolution

If the data file is large enough to make the plotted lines difficult to read, the plot can be split into multiple sections with better resolution.

Steps:
1. Get to the DATA_SPLITS constant at the beggining of the data_parser.py file
2. Change its value to the number of sections in which the plot will be split