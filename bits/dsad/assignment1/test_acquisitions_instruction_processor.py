from unittest import TestCase
from acquisitions_instruction_processor import *

class Test(TestCase):
    def test_process_instructions(self):
        list =[ "Company: ce",
                "No of operations: 6",
                "DETAIL ce",
                "ACQUIRED:aviation BY:ce",
                "ACQUIRED:power BY:ce",
                "ACQUIRED:healthcare BY:ce",
                "DETAIL ce",
                "ACQUIRED:additive BY:aviation",
                "ACQUIRED:additive BY:aviation",
                "DETAIL aviation",
                "RELEASE additive",
                "RELEASE additive",
                "ACQUIRED:additive BY:aviation",
                "ACQUIRED:wind-energy BY:power",
                "ACQUIRED:solar-energy BY:power",
                "ACQUIRED:appliances BY:power",
                "ACQUIRED:ct-manufacturer BY:healthcare",
                "ACQUIRED:lifescience BY:healthcare",
                "ACQUIRED:pharma BY:healthcare",
                "DETAIL ce",
                "DETAIL aviation",
                "DETAIL power",
                "DETAIL lifescience"]
        result = process_instructions(list)
        print(f"result={result}")


