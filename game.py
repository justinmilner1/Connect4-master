import numpy as np
from termcolor import colored
''' 
    Changes for connect4
    -* convert board to 2d
    -* play() --> how the played piece falls to the lowest available row.
            --> invalid move if specific column played on is full
    -* game_status() --> winning options to be updated
    -* print_board() --> update to account for 6x7 rather than 3x3
'''

class Game:
    """
    Connect4 game class
    """
    board = np.zeros((6, 7))
    current_player = 1  # first player is 1, second player is -1
    player1 = None
    player2 = None

    _invalid_move_played = False

    def __init__(self, player1, player2, p1_name, p2_name,
                 winning_reward=1,
                 losing_reward=-1,
                 tie_reward=0,
                 invalid_move_reward=-10):
        self.player1 = player1
        self.player2 = player2
        self.player1.name = p1_name
        self.player2.name = p2_name
        self.player1.player_id = 1
        self.player2.player_id = -1
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.invalid_move_reward = invalid_move_reward
        self.tie_reward = tie_reward
        self.reset()

    @property
    def active_player(self):
        if self.current_player == 1:
            return self.player1
        else:
            return self.player2

    @property
    def inactive_player(self):
        if self.current_player == -1:
            return self.player1
        else:
            return self.player2

    def reset(self):
        self.board = np.zeros((6, 7))
        self.current_player = 1
        self._invalid_move_played = False

    def play(self, column):
        '''
        parameter used to be cell. Now must be column

        '''
        self._invalid_move_played = False
        if column < 0 or column > 6 or len(np.where(self.board[:,column] == 0)[0]) <= 0: # if choice is outside bounds or if column is full
                self.board[0, 0] = 2
                status = self.game_status()
                self._invalid_move_played = True
                #print("Invalid move played at column: ", column)
                return {'winner': status['winner'],
                        'game_over': status['game_over'],
                        'invalid_move': True}
        else:
            #print("playing on column ", column)
            for row in range(5,-1, -1):
                if self.board[row, column] == 0:
                    self.board[row, column] = self.current_player
                    status = self.game_status()
                    #print("player ", self.current_player, " played on ", row, " ", column)
                    return {'winner': status['winner'],
                            'game_over': status['game_over'],
                            'invalid_move': False}


        ####################### old is below
        # self._invalid_move_played = False
        # if cell[0] < 0 or cell[1] < 0 or \
        #         cell[0] >= len(self.board[0]) or cell[1] >= len(self.board[1]) or \
        #             self.board[cell[0], cell[1]] != 0:         #invalid move
        #     self.board[0, 0] = 2
        #     status = self.game_status()
        #     self._invalid_move_played = True
        #     #print("invalid move played: ", cell)
        #     return {'winner': status['winner'],
        #             'game_over': status['game_over'],
        #             'invalid_move': True}
        # else:                               #regular move
        #     self.board[cell[0], cell[1]] = self.current_player
        #     status = self.game_status()
        #     return {'winner': status['winner'],
        #         'game_over': status['game_over'],
        #         'invalid_move': False}

    def next_player(self):
        if not self._invalid_move_played:
            self.current_player *= -1

    def game_status(self):
        winner = 0
        if len(np.where(self.board == 0)[0]) <= 0:   #tie
            winner = 0
            game_over = True
            return {'game_over': game_over, 'winner': winner,
                    'winning_seq': None, 'board': self.board, 'invalid_move': False}
        elif self.board[0, 0] == 2:                    #invalid move
            winner = self.current_player * -1
            game_over = True
            return {'game_over': game_over, 'winner': winner,
                'winning_seq': None, 'board': self.board, 'invalid_move': True}

        winner, winning_seq = self.check_for_win(self.board)

        game_over = winner != 0 or len(list(filter(lambda z: z==0, self.board.flatten()))) == 0
        return {'game_over': game_over, 'winner': winner,
                'winning_seq': winning_seq, 'board': self.board, 'invalid_move': False}

    def check_for_win(self, board):
        # Check horizontal locations for win
        for c in range(0,4):
            for r in range(0, 6):
                if abs(board[r, c] + board[r, c+1] + board[r, c+2] + board[r, c+3]) == 4:
                    return board[r, c], [[r, c], [r, c+1], [r, c+2], [r, c+3]]

        # Check vertical locations for win
        for c in range(0, 7):
            for r in range(0, 3):
                if abs(board[r, c] + board[r+1, c] + board[r+2, c] + board[r+3, c]) == 4:
                    return board[r, c], [[r, c], [r+1, c], [r+2, c], [r+3]]

        # Check positively sloped diaganols
        for c in range(0, 4):
            for r in range(3, 6):
                if abs(board[r, c] + board[r-1, c+1] + board[r-2, c+2] + board[r-3, c+3]) == 4:
                    return board[r, c], [[r, c], [r-1, c+1], [r-2, c+2], [r-3, c+3]]

        # Check negatively sloped diaganols
        for c in range(0, 4):
            for r in range(0, 3):
                if abs(board[r, c] + board[r+1, c+1] + board[r+2, c+2] + board[r+3, c+3]) == 4:
                    return board[r, c], [[r, c], [r+1, c+1], [r+2, c+2], [r+3, c+3]]

        return False, [None]

        # winning_options = [[0, 1, 2, 4], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]
        #                     [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12, 13], [11, 12, 13, 14]
        #                    [14, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19], [17, 18, 19, 20], [18, 19, 20, 21]
        #                    [21, 22, 23, 24], [22, 23, 24, 25], [23, 24, 25, 26], [24, 25, 26, 27], [25, 26, 27, 28]
        #                    [28, 29, 30, 31], [29, 30, 31, 32], [30, 31, 32, 33], [31, 32, 33, 34], [32, 33, 34, 35]
        #                    [35, 36, 37, 38], [36, 37, 38, 39], [37, 38, 39, 40], [38, 39, 40, 41],          #horizontals
        #     [0, 7, 14, 21], [7, 14, 21, 28], [14, 21, 28, 35], [1, 8, 15, 22], [8, 15, 22, 29], [15, 22, 29, 36],
        #      [2, 9, 16, 23], [9, 16, 23, 30], [16, 23, 30, 37], [3, 10, 17, 24], [10, 17, 24, 31], [17, 24, 31, 38],
        #      [4, 11, 18, 25], [11, 18, 25, 32], [18, 25, 32, 39], [5, 12, 19, 26], [12, 19, 26, 33], [19, 26, 33, 40],
        #      [6, 13, 20, 27], [13, 20, 27, 34], [20, 27, 34, 41], verticals



    def print_board(self):
        row = ' '
        status = self.game_status()
        for row_num in range(0, 6):
            for column in range(0, 7):
                if self.board[row_num, column] == 1:
                    cell = colored('x', 'red')
                elif self.board[row_num, column] == -1:
                    cell = colored('o', 'blue')
                else:
                    cell = ' '
                row += cell + ' '

                #if row_num % 7 != 0:
                row += '| '

            print(row)
            if row_num < 5:
                print('---------------------------')
            row = ' '
        print('----------------------------')
        print(' 0   1   2   3   4   5   6  ')
        # row = ' '
        # status = self.game_status()
        # rows = []
        # for i in reversed(range(42)):
        #     if self.board[i] == 1:
        #         cell = 'x'
        #     elif self.board[i] == -1:
        #         cell = 'o'
        #     else:
        #         cell = ' '
        #         #cell = str(i)
        #     if status['winner'] != 0 and i in status['winning_seq']:
        #         cell = cell.upper()
        #     row += cell + ' '
        #     if i % 7 != 0:
        #         row += '| '
        #     else:
        #         row = row[::-1]
        #         rows.append(row)
        #         row = ' '
        # for index in range(len(rows) - 1, -1, -1): #print the rows in the reverse order
        #     print(rows[index])
        #     if index != 0:
        #         print('-----------')

