import numpy as np
import pathlib, datetime, torch
from .abstract_game import AbstractGame

# Token definitions for board cells.
EMPTY = 0
FOOTBALL = 1
STONE = 2

# Allowed jump directions (vertical, horizontal, diagonal).
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_JUMP_COUNT = 6
PASS_ACTION_INDEX = 49 + (len(DIRECTIONS) * MAX_JUMP_COUNT)  # 49+48 = 97

def move_to_index(move):
    if move["type"] == "place":
        r, c = move["position"]
        return r * 7 + c
    elif move["type"] == "jump":
        try:
            direction_index = DIRECTIONS.index(move["direction"])
        except ValueError:
            raise ValueError("Invalid jump direction")
        jump_count = len(move["jumped"])
        if jump_count > MAX_JUMP_COUNT:
            jump_count = MAX_JUMP_COUNT
        return 49 + (direction_index * MAX_JUMP_COUNT) + (jump_count - 1)
    elif move["type"] == "pass":
        return PASS_ACTION_INDEX
    else:
        raise ValueError("Unknown move type")

def index_to_move(index, game_state):
    board = game_state.board_state.board
    if index < 49:
        r = index // 7
        c = index % 7
        return {"type": "place", "position": (r, c)}
    elif index == PASS_ACTION_INDEX:
        return {"type": "pass"}
    else:
        relative = index - 49
        direction_index = relative // MAX_JUMP_COUNT
        jump_count = (relative % MAX_JUMP_COUNT) + 1
        desired_direction = DIRECTIONS[direction_index]
        football_pos = game_state.board_state.find_football()
        legal_jump_moves = get_jump_moves(football_pos, board, game_state.turn)
        for move in legal_jump_moves:
            if move["direction"] == desired_direction and len(move["jumped"]) == jump_count:
                return move
        return None

class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (1, 7, 7)
        self.action_space = list(range(98))
        self.players = [0, 1]
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "expert"

        ### Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 100
        self.num_simulations = 50
        self.discount = 1
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"
        self.support_size = 10

        self.downsample = False
        self.blocks = 3
        self.channels = 64
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 4
        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers = [64]
        self.resnet_fc_policy_layers = [64]

        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = []
        self.fc_policy_layers = []

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parent.parent / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.005
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 10
        self.td_steps = 10
        self.PER = True
        self.PER_alpha = 0.5

        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def visit_softmax_temperature_fn(self, trained_steps):
        return 1

class BoardState:
    """
    Represents the board configuration (7x7). The football starts at the center.
    """
    def __init__(self, m, n, board=None):
        self.m = m
        self.n = n
        self.width = 2 * m + 1
        self.height = 2 * n + 1
        if board is None:
            self.board = np.zeros((self.height, self.width), dtype=int)
            self.board[n, m] = FOOTBALL
        else:
            self.board = board.copy()

    def copy(self):
        return BoardState(self.m, self.n, board=self.board)

    def find_football(self):
        pos = np.argwhere(self.board == FOOTBALL)
        if pos.shape[0] == 0:
            return None
        return tuple(pos[0])

class GameState:
    """
    Holds the complete game state.
    """
    def __init__(self, board_state, turn=0, jump_mode=False):
        self.board_state = board_state
        self.turn = turn  # now using 0 or 1
        self.jump_mode = jump_mode

    def copy(self):
        return GameState(self.board_state.copy(), self.turn, self.jump_mode)

def get_jump_moves(football_pos, board, player):
    """
    For the football at football_pos, return a list of jump moves.
    Each move is a dictionary with:
      - type: "jump"
      - direction: (dr, dc)
      - jumped: list of positions jumped over
      - landing: (row, col) where the football will land
    """
    jump_moves = []
    r0, c0 = football_pos
    height, width = board.shape

    for dr, dc in DIRECTIONS:
        r, c = r0 + dr, c0 + dc
        if c < 0 or c >= width:
            continue
        if r < 0 or r >= height or board[r, c] != STONE:
            continue

        jumped = []
        while 0 <= r < height and 0 <= c < width and board[r, c] == STONE:
            jumped.append((r, c))
            r += dr
            c += dc

        landing_r = r0 + (len(jumped) + 1) * dr
        landing_c = c0 + (len(jumped) + 1) * dc

        if landing_c < 0 or landing_c >= width:
            continue
        if 0 <= landing_r < height:
            if board[landing_r, landing_c] != EMPTY:
                continue
        move = {
            "type": "jump",
            "direction": (dr, dc),
            "jumped": jumped,
            "landing": (landing_r, landing_c)
        }
        jump_moves.append(move)
    return jump_moves

def get_moves(game_state):
    """
    Given a GameState, return a list of legal moves.
    When jump_mode is True, only jump moves (and pass) are allowed.
    Otherwise, include placement moves, jump moves, and pass.
    """
    legal_moves = []
    board = game_state.board_state.board
    height, width = board.shape
    football_pos = game_state.board_state.find_football()

    jump_moves = get_jump_moves(football_pos, board, game_state.turn)
    if game_state.jump_mode:
        legal_moves.extend(jump_moves)
    else:
        for r in range(height):
            for c in range(width):
                if board[r, c] == EMPTY:
                    legal_moves.append({"type": "place", "position": (r, c)})
        legal_moves.extend(jump_moves)
    legal_moves.append({"type": "pass"})
    return legal_moves

def check_win(game_state):
    """
    Check if a winning jump move is available.
    Winning conditions:
      - Player 0 wins if the football reaches (or goes beyond) the bottom row.
      - Player 1 wins if the football reaches (or goes above) the top row.
    """
    board = game_state.board_state.board
    height, _ = board.shape
    football_pos = game_state.board_state.find_football()
    jump_moves = get_jump_moves(football_pos, board, game_state.turn)
    for move in jump_moves:
        landing_r, _ = move["landing"]
        if game_state.turn == 0 and landing_r >= height - 1:
            return True
        if game_state.turn == 1 and landing_r <= 0:
            return True
    return False

class Game(AbstractGame):
    """
    A MuZero-compatible game wrapper for the Conway-Putball/Phutball-inspired game.
    """
    def __init__(self, seed=None):
        self.m = 3
        self.n = 3
        self.reset()

    def reset(self):
        board_state = BoardState(self.m, self.n)
        self.game_state = GameState(board_state, turn=0, jump_mode=False)
        return self.get_observation()

    def step(self, action):
        move = index_to_move(action, self.game_state)
        if move is None:
            move = {"type": "pass"}
        if move["type"] == "place":
            r, c = move["position"]
            if self.game_state.board_state.board[r, c] == EMPTY:
                self.game_state.board_state.board[r, c] = STONE
            self.game_state.jump_mode = False
            self.game_state.turn = 1 if self.game_state.turn == 0 else 0
        elif move["type"] == "jump":
            football_pos = self.game_state.board_state.find_football()
            landing_r, landing_c = move["landing"]
            for (r, c) in move["jumped"]:
                self.game_state.board_state.board[r, c] = EMPTY
            r0, c0 = football_pos
            self.game_state.board_state.board[r0, c0] = EMPTY
            if 0 <= landing_r < self.game_state.board_state.height:
                self.game_state.board_state.board[landing_r, landing_c] = FOOTBALL
            self.game_state.jump_mode = True  # Remain in jump mode.
            # Do not switch turn.
        elif move["type"] == "pass":
            if self.game_state.jump_mode:
                self.game_state.jump_mode = False
            self.game_state.turn = 1 if self.game_state.turn == 0 else 0

        done = check_win(self.game_state)
        reward = 1 if done else 0
        return self.get_observation(), reward, done

    def get_observation(self):
        board = self.game_state.board_state.board.copy()
        return np.array([board])

    def render(self):
        print("Current Board:")
        print(self.game_state.board_state.board)
        if self.game_state.turn == 0:
            print("Turn: Player 1 (Aim for the bottom row – high row numbers)")
        else:
            print("Turn: Player 2 (Aim for the top row – low row numbers)")
        if self.game_state.jump_mode:
            print("Jump Mode: Only jump moves or pass are allowed.")

    def human_to_action(self):
        self.render()
        legal = self.legal_actions()
        print("Legal moves:")
        for idx, a in enumerate(legal):
            move = index_to_move(a, self.game_state)
            print(f"{idx}: {move}")
        choice = int(input("Enter the move number: "))
        return legal[choice]

    def legal_actions(self):
        moves = get_moves(self.game_state)
        return [move_to_index(m) for m in moves]

    def action_to_string(self, action):
        move = index_to_move(action, self.game_state)
        return str(move)

    # --- NEW: Override to_play() so that interactive play always happens ---
    def to_play(self):
        # Force interactive play by always returning 1.
        return self.game_state.turn

if __name__ == '__main__':
    game = Game()
    print("Initial observation:")
    print(game.get_observation())
    game.render()
    print("\nLegal moves:")
    for a in game.legal_actions():
        print(a)

