## This module reads and writes the files ##

def read_from_file(input_file_name: str) -> list:
    """
    This funtion reads the file contents adds to a list and returns the list.  Each item in a list is considered as
    a line
    It throws File not found exception if the file doesnt exist
    :param input_file_name:
    :return:
    """

    try:
        with open(input_file_name, "r") as in_file:
            lines = in_file.readlines()
    except IOError as io:
        print(f"Exception while opening the file {input_file_name}")
        raise io

    return lines

def write_to_file(output_file_name:str, output_lines:list):
    """
    This function reads all the lies from the list and writes to a file
    Throws an exception if the file doesnt exist
    :param output_file_name:
    :param output_lines:
    :return:
    """
    try:
        with open(output_file_name, "w") as output_file:
            output_file.writelines(output_lines)
    except IOError as io:
        print(f"Exception while opening the file {output_file_name}")
        raise io
