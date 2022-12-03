import numpy as np
from random import choices
NUM_COMBOS = 10000


for combo_length in range(1, 6):
    for desired_total in range(combo_length, combo_length*3+1):

        valid_combos = []

        while len(valid_combos) < NUM_COMBOS:
            combo = choices([1, 2, 3], k=combo_length)
            if sum(combo) == desired_total:
                valid_combos.append(combo)

        valid_combos = np.array(valid_combos)
        print("(%d, %d): [%f, %f, %f]" % (combo_length, desired_total,
            np.count_nonzero(valid_combos == 1) / NUM_COMBOS / combo_length,
            np.count_nonzero(valid_combos == 2) / NUM_COMBOS / combo_length,
            np.count_nonzero(valid_combos == 3) / NUM_COMBOS / combo_length,
        ))
        
