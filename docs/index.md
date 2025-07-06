# Welcome to RQEquilibrium Documentation

This is the homepage for the RQEquilibrium library documentation.

This is an implementation of the Risk-Averse Quantal Equilibrium (RQE) solution concepts. Which was introduced in the paper: Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics and published as conference paper at ICLR 2025.

If you want to cite the orinal paper, please use the following BibTeX entry:

```bibtex
@inproceedings{
mazumdar2025tractable,
title={Tractable Multi-Agent Reinforcement Learning through Behavioral Economics},
author={Eric Mazumdar, Kishan Panaganti, Laixi Shi},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=stUKwWBuBm}
}
```
## Usage

Here's a basic example of how to use the library:

```python
from rqequilibrium import RQE, Player

R1 = np.array([[200, 160], [370, 10]])
R2 = np.array([[160, 10], [200, 370]])
R3 = R1.copy()
R4 = R2.copy()
RQE.print_game(R1, R2)
players = [
    Player(tau=0.001, epsilon=170, game_matrix=R1),
    Player(tau=0.06, epsilon=110, game_matrix=R2),
    Player(tau=0.003, epsilon=190, game_matrix=R3),
    Player(tau=0.05, epsilon=130, game_matrix=R4),
]
rqe_solver = RQE(players=players, lr=1e-4, max_iter=1000, br_iters=50)

print("Computed policies for RQE")
# print("Player 1:", np.argmax(pi1), "with policy:", pi1)
i = 0
for pi in rqe_solver.optimize():
    print(f"Player {i + 1}: Best Action {np.argmax(pi)} with policy: {pi}")
    i += 1

```

## TODOs

- [x] Implementing a simple RQE solver.
    * [x] Implementing the D functions and v functions implemented in the paper.
    * [x] Implementing the Projected Gradient Descent.
- [ ] Testing the RQE solver.
    * [ ] Testing using the original Paper Clif Walking.
    * [x] Testing using the original Paper 13 games matrix.
    * [ ] Testing using Google's Deepmind OpenSpiel library.
    * [x] Testing the n player games.
- [ ] Adding more features.
    * [x] Working for n player games.
    * [ ] Using JAX for faster computation on bigger games.

