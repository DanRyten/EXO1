import os

def remove_undefined_lines(file_path):
    with open(file_path, 'r+') as file:
        # Move the cursor to the end of the file
        file.seek(0, os.SEEK_END)
        
        last_four_line_positions = []
        lines_to_remove = 0
        position = file.tell()
        
        # Start reading the file backwards
        while position >= 0 and lines_to_remove < 4:
            file.seek(position)
            char = file.read(1)
            if char == '\n':
                lines_to_remove += 1
                last_four_line_positions.append(position)
            position -= 1
        
        # If last two lines contain 'undefined' in the second column, truncate the file
        if lines_to_remove == 4:
            # Read the last two lines
            file.seek(last_four_line_positions[-1] + 1)
            last_line = file.readline()
            file.seek(last_four_line_positions[-2] + 1)
            second_last_line = file.readline()
            file.seek(last_four_line_positions[-3] + 1)
            third_last_line = file.readline()
            file.seek(last_four_line_positions[-4] + 1)
            fourth_last_line = file.readline()
            
            # Check if 'undefined' exists in the second column of both lines
            if 'undefined;' in second_last_line and 'undefined;' in last_line:
                # Truncate the file
                file.truncate(last_four_line_positions[-1])

def process_folder_files(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            remove_undefined_lines(file_path)


folder_path = '/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/'
process_folder_files(folder_path)
