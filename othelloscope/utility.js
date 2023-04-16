const eights = [
  [0, 1],
  [1, 0],
  [-1, 0],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

class OthelloBoardState {
  constructor(board_size = 8) {
    this.board_size = board_size * board_size;
    let board = Array.from({ length: 8 }, () => new Array(8).fill(0));
    board[3][4] = 1;
    board[3][3] = -1;
    board[4][3] = 1;
    board[4][4] = -1;
    this.initial_state = board;
    this.state = this.initial_state;
    this.age = Array.from({ length: 8 }, () => new Array(8).fill(0));
    this.next_hand_color = 1;
    this.history = [];
  }

  get_occupied() {
    let board = this.state;
    let tbr = board.flat().map((x) => x !== 0);
    return tbr;
  }

  get_state() {
    let board = this.state.map((row) => row.map((x) => x + 1));
    let tbr = board.flat();
    return tbr;
  }

  get_age() {
    return this.age.flat();
  }

  get_next_hand_color() {
    return this.next_hand_color + 1; // 2;
  }

  update(moves, prt = false) {
    if (prt) {
      this.__print__();
    }
    for (let move of moves) {
      this.umpire(move);
      if (prt) {
        this.__print__();
      }
    }
  }

  umpire(move) {
    let r = Math.floor(move / 8),
      c = move % 8;
    if (this.state[r][c] !== 0) {
      throw new Error(`${r}-${c} is already occupied!`);
    }
    let occupied = this.state.flat().filter((x) => x !== 0).length;
    let color = this.next_hand_color;
    let tbf = [];
    for (let direction of eights) {
      let buffer = [];
      let cur_r = r,
        cur_c = c;
      while (true) {
        cur_r += direction[0];
        cur_c += direction[1];
        if (cur_r < 0 || cur_r > 7 || cur_c < 0 || cur_c > 7) {
          break;
        }
        if (this.state[cur_r][cur_c] === 0) {
          break;
        } else if (this.state[cur_r][cur_c] === color) {
          tbf.push(...buffer);
          break;
        } else {
          buffer.push([cur_r, cur_c]);
        }
      }
    }
    if (tbf.length === 0) {
      color *= -1;
      this.next_hand_color *= -1;
      for (let direction of eights) {
        let buffer = [];
        let cur_r = r,
          cur_c = c;
        while (true) {
          cur_r += direction[0];
          cur_c += direction[1];
          if (cur_r < 0 || cur_r > 7 || cur_c < 0 || cur_c > 7) {
            break;
          }
          if (this.state[cur_r][cur_c] === 0) {
            break;
          } else if (this.state[cur_r][cur_c] === color) {
            tbf.push(...buffer);
            break;
          } else {
            buffer.push([cur_r, cur_c]);
          }
        }
      }
    }
    if (tbf.length === 0) {
      let valids = this.get_valid_moves();
      if (valids.length === 0) {
        throw new Error("Both color cannot put piece, game should have ended!");
      } else {
        throw new Error("Illegal move!");
      }
    }
    this.age = this.age.map((row) => row.map((x) => x + 1));
    for (let ff of tbf) {
      this.state[ff[0]][ff[1]] *= -1;
      this.age[ff[0]][ff[1]] = 0;
    }
    this.state[r][c] = color;
    this.age[r][c] = 0;
    this.next_hand_color *= -1;
    this.history.push(move);
  }

  __print__() {
    console.log("-".repeat(20));
    console.log(this.history.map(permit_reverse));
    let a = "abcdefgh";
    for (let k = 0; k < this.state.length; k++) {
      let row = this.state[k];
      let tbp = [];
      for (let ele of row) {
        if (ele === -1) {
          tbp.push("O");
        } else if (ele === 0) {
          tbp.push(" ");
        } else {
          tbp.push("X");
        }
      }
      console.log([a[k], ...tbp].join(" "));
    }
    let tbp = Array.from({ length: 8 }, (_, i) => i + 1).map(String);
    console.log([" ", ...tbp].join(" "));
    console.log("-".repeat(20));
  }

  tentative_move(move) {
    let r = Math.floor(move / 8),
      c = move % 8;
    if (this.state[r][c] !== 0) {
      return 0;
    }
    let occupied = this.state.flat().filter((x) => x !== 0).length;
    let color = this.next_hand_color;
    let tbf = [];
    for (let direction of eights) {
      let buffer = [];
      let cur_r = r,
        cur_c = c;
      while (true) {
        cur_r += direction[0];
        cur_c += direction[1];
        if (cur_r < 0 || cur_r > 7 || cur_c < 0 || cur_c > 7) {
          break;
        }
        if (this.state[cur_r][cur_c] === 0) {
          break;
        } else if (this.state[cur_r][cur_c] === color) {
          tbf.push(...buffer);
          break;
        } else {
          buffer.push([cur_r, cur_c]);
        }
      }
    }
    if (tbf.length !== 0) {
      return 1;
    } else {
      color *= -1;
      for (let direction of eights) {
        let buffer = [];
        let cur_r = r,
          cur_c = c;
        while (true) {
          cur_r += direction[0];
          cur_c += direction[1];
          if (cur_r < 0 || cur_r > 7 || cur_c < 0 || cur_c > 7) {
            break;
          }
          if (this.state[cur_r][cur_c] === 0) {
            break;
          } else if (this.state[cur_r][cur_c] === color) {
            tbf.push(...buffer);
            break;
          } else {
            buffer.push([cur_r, cur_c]);
          }
        }
      }
      if (tbf.length === 0) {
        return 0;
      } else {
        return 2;
      }
    }
  }

  get_valid_moves() {
    let regular_moves = [];
    let forfeit_moves = [];
    for (let move = 0; move < 64; move++) {
      let x = this.tentative_move(move);
      if (x === 1) {
        regular_moves.push(move);
      } else if (x === 2) {
        forfeit_moves.push(move);
      }
    }
    if (regular_moves.length) {
      return regular_moves;
    } else if (forfeit_moves.length) {
      return forfeit_moves;
    } else {
      return [];
    }
  }

  get_gt(moves, func, prt = false) {
    let container = [];
    if (prt) {
      this.__print__();
    }
    for (let move of moves) {
      this.umpire(move);
      container.push(this[func]());
      if (prt) {
        this.__print__();
      }
    }
    return container;
  }
}

function permit_reverse(_) {
  // Implement permit_reverse function if needed
}
