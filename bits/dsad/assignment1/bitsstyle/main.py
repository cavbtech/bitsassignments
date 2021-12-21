import enum

## This module have methods to add / delete a node to a general tree and also travese methods
class GeneralTreeNode:
    """
     GeneralTree is an implementation construct a general tree with add, delete and traverse methods
    """
    def __init__(self, parent_node_data_item):
        """
        General tree implementation.  Each node holds its data and all children it has
        :param parent_node_data_item:
        """
        self.parent_node_data_item      = parent_node_data_item
        self.children                   = []


    def find_company(self, company_name: str):
        """
        find_company recursively finds the comany name
        :param company_name:
        :return:
        """
        # print(f"x={company_name} and self.parent_node_data_item={self.parent_node_data_item}")
        if (company_name == self.parent_node_data_item):
            return self
        if self.children:
            for child in self.children:
                if child.parent_node_data_item == company_name:
                    return child
                child.find_company(company_name)


    def tree_traverse(self,root):
        """ This method follows level order traversal code
        :param root:
        :return:
        """
        if (root == None):
            return;

        q = []  # Create a queue
        q.append(root);  # Enqueue root
        while (len(q) != 0):
            n = len(q);
            # If this node has children
            while (n > 0):
                # Dequeue an item from queue and print it
                p = q[0]
                q.pop(0);
                print(p.parent_node_data_item, end=' ')
                # Enqueue all children of the dequeued item
                for i in range(len(p.children)):
                    q.append(p.children[i]);
                n -= 1
            print()  # Print new line between two levels

class COMPANY_ALREADY_EXIST(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class COMPANY_NAME_CANNOT_BE_NONE(Exception):
    def __init__(self):
        self.message = "company name cannot be None"
        super().__init__(self.message)

class COMPANY_DOESNT_EXIST(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class Company:

    def __init__(self, grand_company_name):
        self.parent_comapny             = GeneralTreeNode(grand_company_name)
        self.all_comapny_names_cache    = {grand_company_name}
        self.trace_log                  = []




    def detail(self,company_name):
        """ This function prints the parent and immediate children of company"""
        # print("before detail")
        if company_name == None:
            raise COMPANY_NAME_CANNOT_BE_NONE

        parent_node: GeneralTreeNode = self.parent_comapny.find_company(company_name)
        if parent_node == None:
            self.trace_log.append(f"DETAIL:{company_name}")
            self.trace_log.append(f"Acquired companies:None")
            self.trace_log.append(f"No of companies acquired:0")
            raise COMPANY_DOESNT_EXIST(f"""company {company_name} doesnt exist""")
        # print(f"parent_node={parent_node.parent_node_data_item} and childern={parent_node.children}")
        if len(parent_node.children)>0:
            acquired_company_names = ", ".join(list(map(lambda x:x.parent_node_data_item, parent_node.children)))
        else:
            acquired_company_names  = None
        self.trace_log.append(f"DETAIL:{company_name}")
        self.trace_log.append(f"Acquired companies:{acquired_company_names}")
        self.trace_log.append(f"No of companies acquired:{len(parent_node.children)}")




    def acquire(self,parent_company, acquired_company):
        # print("before acquire")
        """Inserts the acquired_company as a new child node to the parent_company"""
        if acquired_company in self.all_comapny_names_cache:
            self.trace_log.append(f"""ACQUIRED FAILED: {acquired_company} BY: {parent_company}""")
            raise COMPANY_ALREADY_EXIST(f"""company {acquired_company} already acquired""")
        parent_node:GeneralTreeNode = self.parent_comapny.find_company(parent_company)
        if parent_node == None:
            self.trace_log.append(f"""ACQUIRED FAILED: {acquired_company} BY: {parent_company}""")
            raise COMPANY_DOESNT_EXIST(f"""company {parent_company} doesnt exist""")
        parent_node.children.append(GeneralTreeNode(acquired_company))
        self.all_comapny_names_cache.add(acquired_company)
        self.trace_log.append(f"ACQUIRED SUCCESS:{parent_company} succesfully acquired {acquired_company}")

    def release(self, company_tobe_released):
        """removes the node mentioned in the released_company"""
        #print(f"before release {company_tobe_released} {self.all_comapny_names_cache} and {company_tobe_released in self.all_comapny_names_cache}")
        if (company_tobe_released in self.all_comapny_names_cache) == False:
            self.trace_log.append(f"""RELEASED FAILED: released {company_tobe_released} failed""")
            raise COMPANY_DOESNT_EXIST(f"""company {company_tobe_released} doesnt exist""")
            return

        company_tobe_released_node:GeneralTreeNode = self.parent_comapny.find_company(company_tobe_released)
        if company_tobe_released_node == None:
            self.trace_log.append(f"""RELEASED FAILED: released {company_tobe_released} failed""")
            raise COMPANY_DOESNT_EXIST(f"""company {company_tobe_released} doesnt exist""")
        parent_node:GeneralTreeNode             = self.parent_comapny.find_company(company_tobe_released_node.parent_node_data_item)

        self.remove_child(parent_node,company_tobe_released)
        #print(f"self.all_comapny_names_cache={self.all_comapny_names_cache}")
        self.all_comapny_names_cache.remove(company_tobe_released)
        self.trace_log.append(f"RELEASED SUCCESS: released {company_tobe_released} successfully")

    def remove_child(self,parent:GeneralTreeNode,company_tobe_released):
        """
        This method releases the child from the parent node by identifying the child index
        :param parent:
        :param child:
        :return:
        """
        ##TODO: can be improved by using any other search.  This is sequential search
        i =-1;
        existFlag = False
        for child in parent.children:
            i += 1
            # print(f"child.parent_node_data_item={child.parent_node_data_item} and "
            #       f"company_tobe_released = {company_tobe_released}")
            if child.parent_node_data_item == company_tobe_released:
                existFlag = True
                break
        if(existFlag):
            parent.children.pop(i)

"""
acquisitions instructor interpreter reads the list of acquisition, release and detail instructions given in the file
"""
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
    except COMPANY_ALREADY_EXIST as e:
        print(e)

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
        try:
            company.release(instruction_split[1].strip())
        except COMPANY_DOESNT_EXIST as e:
            print(e)
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
        try:
            company.detail(instruction_split[1].strip())
        except COMPANY_DOESNT_EXIST as e:
            print(e)
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
            for outline in output_lines:
                output_file.write(outline+'\n')
    except IOError as io:
        print(f"Exception while opening the file {output_file_name}")
        raise io

if __name__ == '__main__':
    input_instructions   = read_from_file("./inputPS5.txt")
    trace_logs_list      = process_instructions(input_instructions)
    write_to_file('./outputPS5.txt',trace_logs_list)




