import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from cython import parallel

@cython.cdivision(True)
@cython.nogil
@cython.cfunc
@cython.exceptval(check=False)
def random() -> cython.double:
    x = cython.declare(cython.double, rand())  
    return x / RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def find_valid_bombs(probabilities: cython.double[:,::1], board: cython.int[:, ::1], row_constraint: cython.int[::1] , col_constraint: cython.int[::1] , num_boards: cython.int):

    threads = 16
    final_boards_np = np.zeros((num_boards, 5, 5), int)
    final_boards: cython.int[:,:,::1] = final_boards_np
    row_bomb_count: cython.int[:,::1] = np.zeros((threads,5), dtype=np.int)
    col_bomb_count: cython.int[:,::1] = np.zeros((threads,5), dtype=np.int)
    cur_board: cython.int[:,:,::1] = np.zeros((threads,5,5), dtype=np.int)
    board_num = cython.declare(cython.Py_ssize_t)

    for board_num in parallel.prange(num_boards, schedule='guided', nogil="true", num_threads=threads):
        worker_num = cython.declare(cython.int)
        worker_num = parallel.threadid()
        fail = cython.declare(cython.int, 1)
        x = cython.declare(cython.Py_ssize_t)
        y = cython.declare(cython.Py_ssize_t)
        while fail:
            fail = False

            # Reset the board
            for x in range(5):
                for y in range(5):
                    cur_board[worker_num,y,x] = board[y, x]

            for x in range(5):
                col_bomb_count[worker_num,x] = 0

            for y in range(5):
                row_bomb_count[worker_num, y] = 0
                for x in range(5):
                    if cur_board[worker_num,y,x] == -1:
                        cur_board[worker_num,y,x] = (random() < probabilities[y, x])
                    if cur_board[worker_num,y,x] == 1:
                        row_bomb_count[worker_num, y] += 1
                        if row_bomb_count[worker_num, y] > row_constraint[y]:
                            fail = True
                            break
                        col_bomb_count[worker_num, x] += 1
                        if col_bomb_count[worker_num, x] > col_constraint[x]:
                            fail = True
                            break
            
                if row_bomb_count[worker_num, y] != row_constraint[y]:
                    fail = True
                if fail:
                    break

            if fail:
                continue  
        
            for x in range(5):
                if col_bomb_count[worker_num, x] != col_constraint[x]:
                    fail = True
                    break
            
        for x in range(5):
            for y in range(5):
                final_boards[board_num, y, x] = cur_board[worker_num,y,x]
        
    return final_boards_np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_valid_boards(probabilities: cython.double[:,:,::1], bombs: cython.int[:,:,::1], board: cython.int[:, ::1], row_constraint: cython.int[::1], col_constraint: cython.int[::1], num_boards: cython.int):
   
    threads = 16
    final_boards_np = np.zeros((num_boards, 5, 5), int)
    final_boards: cython.int[:,:,::1] = final_boards_np
    col_total: cython.int[:,::1] = np.zeros((threads,5), dtype=np.int)
    row_total: cython.int[:,::1] = np.zeros((threads,5), dtype=np.int)
    cur_board: cython.int[:,:,::1] = np.zeros((threads,5,5), dtype=np.int)
    
    bomb_index = cython.declare(cython.Py_ssize_t, 0)
    board_num = cython.declare(cython.Py_ssize_t)
    

    for board_num in parallel.prange(num_boards, schedule='guided', nogil="true", num_threads=threads):
        
        worker_num = cython.declare(cython.int)
        worker_num = parallel.threadid()
        fail = cython.declare(cython.int, True)
        x = cython.declare(cython.Py_ssize_t)
        y = cython.declare(cython.Py_ssize_t)
        i = cython.declare(cython.Py_ssize_t)
        

        while fail:
            fail = False

            # Reset the board
            for x in range(5):
                for y in range(5):
                    cur_board[worker_num, y, x] = board[y, x]

            bomb_index = (bomb_index + 1) % bombs.shape[0]
            for x in range(5):
                for y in range(5):
                    if bombs[bomb_index,y,x] == 1:
                        cur_board[worker_num, y, x] = 0
            

            for x in range(5):
                col_total[worker_num,x] = 0

            for y in range(5):
                row_total[worker_num,y] = 0
                for x in range(5):
                    if cur_board[worker_num, y, x] == -1:
                        random_number = cython.declare(cython.double, random())
                        for i in range(3):
                            if random_number < probabilities[y, x, i]:
                                cur_board[worker_num, y, x] = i+1
                                break
                            random_number = random_number - probabilities[y, x, i]
                    row_total[worker_num,y] += cur_board[worker_num, y, x]
                    if row_total[worker_num,y] > row_constraint[y]:
                        fail = True
                        break
                    col_total[worker_num,x] += cur_board[worker_num, y, x]
                    if col_total[worker_num,x] > col_constraint[x]:
                        fail = True
                        break
            
                if row_total[worker_num,y] != row_constraint[y]:
                    fail = True
                if fail:
                    break
 
            if fail: 
                continue  
        
            for x in range(5):
                if col_total[worker_num, x] != col_constraint[x]:
                    fail = True
                    break

    
        for x in range(5):
            for y in range(5):
                final_boards[board_num, y, x] = cur_board[worker_num,y, x]
        
    return final_boards_np

