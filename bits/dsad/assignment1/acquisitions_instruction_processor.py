"""
acquisitions instructor interpreter reads the list of acquisition, release and detail instructions given in the file
"""

import enum

from company_operations import *
from file_opeartions import *


class acquisition_instructions(enum.Enum):
    COMPANY  = 0
    DETAIL = 1
    ACQUIRED = 2
    RELEASE = 3
    BY = 4


def do_acquisition(company:Company, instruction:str):
    """
    In case of acquisition the instruction must be in "ACQUIRED:aviation BY:ce" format
    :param company:
    :param instruction_split:
    :return:
    """
    instruction_base_split = instruction.strip().split(" ")
    if len(instruction_base_split) !=2:
        raise Exception(f"""Not a valid acquisition instruction {instruction} 
        it should be in "ACQUIRED:aviation BY:ce" format """)
    try:
        company_acquired       = instruction_base_split[0].split(":")[1]
        company_acquired_by    = instruction_base_split[1].split(":")[1]
    except:
        raise Exception(f"""Not a valid acquisition instruction {instruction} 
                it should be in "ACQUIRED:aviation BY:ce" format """)

    try:
        company.acquire(company_acquired_by.strip(),company_acquired.strip())
    except Exception as e:
        raise e



def do_release(company:Company, instruction:str):
    """
    Release a company instruction should be in the following format "RELEASE additive"
    :param company:
    :param instruction:
    :return:
    """
    instruction_split = instruction.split(" ")
    if(len(instruction_split)>1):
        company.release(instruction[1].strip())
    else:
        raise Exception(f"""Not a valid release instuction {instruction}. 
        Expected instruction as <RELEASE additive> where additive is company name""")



def do_detail(company:Company, instruction:str):
    """
    Details a company instruction should be in the following format "DETAIL ce"
    :param company:
    :param instruction:
    :return:
    """
    instruction_split = instruction.split(" ")
    if (len(instruction_split) > 1):
        company.detail(instruction_split[1].strip())
    else:
        raise Exception(f"""Not a valid instruction for Detail operation {instruction}. 
        Expected instruction as <DETAIL ce> where ce is company name""")

def create_company(instruction:str):
    """
    It is expected that only one parent company is getting created per file and the instruction should be in
    "Company: ce" format
    :param instruction:
    :return:
    """
    instruction_split = instruction.split(":")
    if len(instruction_split) > 0:
        global company
        #grand_parent_company_node = GeneralTreeNode(instruction_split[1].strip())
        company                   = Company(instruction_split[1].strip())
    else:
        raise Exception(f"Not a valid instruction to open a company {instruction}")


def process_instructions(input_instructions:list)->list:
    """
    process_instructions processes all the instructions that are part of the list
    :param input_instructions:
    :return:
    """
    i = 0
    j = 0
    for instruction in input_instructions:
        if(i==0):
            if instruction.upper().startswith(acquisition_instructions.COMPANY.name) == False:
                raise Exception("""The starting instruction must be create company instruction ex: "Company:abc"  
                where abc is a company name""")

        i += 1
        if instruction.upper().startswith(acquisition_instructions.COMPANY.name):
            create_company(instruction)
            j+=1

        elif instruction.upper().startswith(acquisition_instructions.ACQUIRED.name):
            do_acquisition(company, instruction)
            j += 1
        elif instruction.upper().startswith(acquisition_instructions.RELEASE.name):
            do_release(company,instruction)
            j += 1
        elif instruction.upper().startswith(acquisition_instructions.DETAIL.name):
            do_detail(company,instruction)
            j += 1
        elif instruction.startswith("No of operations") == False:
            print(f"not a valid instruction {instruction}. Please check")

    print(f"total instructions given={i} and processed instructions={j}")
    return company.trace_log

if __name__ == '__main__':
    input_instructions   = read_from_file("")
    trace_logs_list      = process_instructions(input_instructions)
    write_to_file(trace_logs_list)
