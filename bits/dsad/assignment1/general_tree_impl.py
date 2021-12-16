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



    def find_company(self,company_name:str):
        """
        find_company recursively finds the comany name
        :param company_name:
        :return:
        """
        # print(f"x={company_name} and self.parent_node_data_item={self.parent_node_data_item}")
        if(company_name==self.parent_node_data_item):
            return self
        for x in self.children:
            if x.parent_node_data_item == company_name:
                # print(f"x={x} and x.parent_node_data_item={x.parent_node_data_item}")
                return x
            return  self.find_company(x.parent_node_data_item)


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
