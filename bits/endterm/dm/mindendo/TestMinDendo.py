from bits.endterm.dm.mindendo.minDendoD import dendo_min

if __name__ == '__main__':
    distP4ict =  {'P1': {'P1': 0.0, 'P2': 0.155, 'P3': 0.101, 'P4': 0.077, 'P5': 0.119, 'P6': 0.264},
     'P2': {'P1': 0.155, 'P2': 0.0, 'P3': 0.098, 'P4': 0.204, 'P5': 0.274, 'P6': 0.254},
     'P3': {'P1': 0.101, 'P2': 0.098, 'P3': 0.0, 'P4': 0.174, 'P5': 0.204, 'P6': 0.309},
     'P4': {'P1': 0.077, 'P2': 0.204, 'P3': 0.174, 'P4': 0.0, 'P5': 0.113, 'P6': 0.222},
     'P5': {'P1': 0.119, 'P2': 0.274, 'P3': 0.204, 'P4': 0.113, 'P5': 0.0, 'P6': 0.334},
     'P6': {'P1': 0.264, 'P2': 0.254, 'P3': 0.309, 'P4': 0.222, 'P5': 0.334, 'P6': 0.0}}
    dendo_min(distP4ict)