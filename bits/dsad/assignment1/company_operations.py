from general_tree_impl import *


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
        print("before detail")
        if company_name == None:
            raise COMPANY_NAME_CANNOT_BE_NONE

        parent_node: GeneralTreeNode = self.parent_comapny.find_company(company_name)
        if parent_node == None:
            self.trace_log.append(f"DETAIL:{company_name}")
            self.trace_log.append(f"Acquired companies:None")
            self.trace_log.append(f"No of companies acquired:0")
            raise COMPANY_DOESNT_EXIST(f"""company {company_name} doesnt exist""")
        print(f"parent_node={parent_node.parent_node_data_item} and childern={parent_node.children}")
        if len(parent_node.children)>0:
            acquired_company_names = ", ".join(list(map(lambda x:x.parent_node_data_item, parent_node.children)))
        else:
            acquired_company_names  = None
        self.trace_log.append(f"DETAIL:{company_name}")
        self.trace_log.append(f"Acquired companies:{acquired_company_names}")
        self.trace_log.append(f"No of companies acquired:{len(parent_node.children)}")




    def acquire(self,parent_company, acquired_company):
        print("before acquire")
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
        print(f"before release {company_tobe_released} {self.all_comapny_names_cache} and {company_tobe_released in self.all_comapny_names_cache}")
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
        print(f"self.all_comapny_names_cache={self.all_comapny_names_cache}")
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
            print(f"child.parent_node_data_item={child.parent_node_data_item} and "
                  f"company_tobe_released = {company_tobe_released}")
            if child.parent_node_data_item == company_tobe_released:
                existFlag = True
                break
        if(existFlag):
            parent.children.pop(i)



