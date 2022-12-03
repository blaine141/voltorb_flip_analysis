import numpy as np
from random import choices
NUM_COMBOS = 10000

probability_data = np.empty((6,6), dtype=object)

for unknown_cells in range(1, 6):
    for desired_total in range(unknown_cells+1):

        valid_combos = []

        while len(valid_combos) < NUM_COMBOS:
            combo = choices([0,1], k=unknown_cells)
            if sum(combo) == desired_total:
                valid_combos.append(combo)

        valid_combos = np.array(valid_combos)
        print("(%d, %d): [%f, %f]," % (unknown_cells, desired_total,
            np.count_nonzero(valid_combos == 0) / NUM_COMBOS / unknown_cells,
            np.count_nonzero(valid_combos == 1) / NUM_COMBOS / unknown_cells
        ))