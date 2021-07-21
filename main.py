import numpy as np
import pickle
from tqdm import tqdm

BOARD_ROWS = 3
BOARD_COLUMNS = 3
BOARD_DEPTH = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLUMNS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    def get_hash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLUMNS * BOARD_ROWS))
        return self.boardHash

    def check_win(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        for i in range(BOARD_COLUMNS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        print(self.board.reshape(BOARD_COLUMNS * BOARD_ROWS))
        diag1_sum = sum(self.board.diagonal())
        print(self.board.diagonal(), sum(self.board.diagonal()))
        diag2_sum = sum(
            [self.board.reshape(BOARD_COLUMNS * BOARD_ROWS)[i] for i in range(2, 8, 2)]
        )
        print(
            [self.board.reshape(BOARD_COLUMNS * BOARD_ROWS)[i] for i in range(2, 8, 2)],
            sum(
                [
                    self.board.reshape(BOARD_COLUMNS * BOARD_ROWS)[i]
                    for i in range(2, 8, 2)
                ]
            ),
        )
        diag_sum_max = max(abs(diag1_sum), abs(diag2_sum))
        if diag_sum_max == 3:
            self.isEnd = True
            if diag1_sum == 3 or diag2_sum == 3:
                return 1
            else:
                return -1

        if len(self.available_positions()) == 0:
            self.isEnd = True
            return 0

        return None

    def available_positions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLUMNS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def update_state(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def give_reward(self):
        result = self.check_win()
        if result == 1:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif result == -1:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0.1)
            self.p2.feed_reward(0.5)

    def reset_board(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLUMNS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def train(self, epochs=1000):
        for epoch in tqdm(range(epochs)):
            for move_number in range(1, 6):
                positions = self.available_positions()
                p1_action = self.p1.choose_action(
                    positions, self.board, self.playerSymbol
                )
                self.update_state(p1_action)
                board_hash = self.get_hash()
                self.p1.add_state(board_hash)

                if move_number > 2:
                    win = self.check_win()
                    if win is not None:
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset_board()
                        break

                positions = self.available_positions()
                p2_action = self.p2.choose_action(
                    positions, self.board, self.playerSymbol
                )
                self.update_state(p2_action)
                board_hash = self.get_hash()
                self.p2.add_state(board_hash)

                if move_number > 2:
                    win = self.check_win()
                    if win is not None:
                        self.give_reward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset_board()
                        break

    def play_human(self):
        while self.isEnd is not True:
            print(self.p1.name)
            print(self.p2.name)
            positions = self.available_positions()
            p1_action = self.p1.choose_action(positions, self.board, self.playerSymbol)
            self.update_state(p1_action)
            self.show_board()
            win = self.check_win()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset_board()
                return self.p2.name  # wrocic tutaj i naprawic
            else:
                positions = self.available_positions()
                p2_action = self.p2.choose_action(positions)

                self.update_state(p2_action)
                self.show_board()
                win = self.check_win()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset_board()
                    return self.p2.name

    def show_board(self):
        print_characters = {1: "x", -1: "o", 0: " "}
        for i in range(0, BOARD_ROWS):
            print(13 * "-")
            out = "| "
            for j in range(0, BOARD_COLUMNS):
                out += print_characters[self.board[i, j]] + " | "
            print(out)
        print(13 * "-")


class Bot:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def get_hash(self, board):
        board_hash = str(board.reshape(BOARD_ROWS * BOARD_COLUMNS))
        return board_hash

    def choose_action(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board_hash = self.get_hash(next_board)
                value = (
                    0
                    if self.states_value.get(next_board_hash) is None
                    else self.states_value.get(next_board_hash)
                )
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def add_state(self, state):
        self.states.append(state)

    def feed_reward(self, reward):
        for state in reversed(self.states):
            if self.states_value.get(state) is None:
                self.states_value[state] = 0
            self.states_value[state] += self.lr * (
                self.decay_gamma * reward - self.states_value[state]
            )
            reward = self.states_value[state]

    def reset(self):
        self.states = []

    def save_policy(self):
        fw = open("policy_" + str(self.name), "wb")
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, "rb")
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def choose_action(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action


class Test:
    def __init__(self, iterations_list: list):
        self.__iterations_list = iterations_list

    @property
    def iterations_list(self):
        return self.__iterations_list

    @iterations_list.setter
    def iteration_list(self, iterations_list: list):
        self.__iterations_list = iterations_list

    def test_with_human(self):
        for iterations_amount in self.__iterations_list:
            p1 = Bot("p1")
            p2 = Bot("p2")
            state = State(p1, p2)
            state.train(iterations_amount)
            p1.save_policy()

            p1 = Bot("Bot", exp_rate=0)
            p1.load_policy("policy_p1")
            p2 = HumanPlayer("Human")
            state = State(p1, p2)
            result = state.play_human()


if __name__ == "__main__":
    # training
    p1 = Bot("p1")
    p2 = Bot("p2")

    st = State(p1, p2)
    print("training...")
    st.train(1000)
    p1.save_policy()

    # play with human
    p1 = Bot("computer", exp_rate=0)
    p1.load_policy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    # while True:
    stan = st.play_human()
