from unittest import TestCase
from acquisitions_instruction_processor import *

class Test(TestCase):
    def test_process_instructions(self):
        list =["Company: ce","No of operations: 6","DETAIL ce","ACQUIRED:aviation BY:ce",
               "ACQUIRED:power BY:ce","DETAIL ce"]
        result = process_instructions(list)
        print(f"result={result}")


