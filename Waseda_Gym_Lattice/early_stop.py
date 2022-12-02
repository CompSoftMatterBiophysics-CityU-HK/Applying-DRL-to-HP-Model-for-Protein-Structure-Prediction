import numpy as np


def delta_OSB_ES(O_S_B, E_S, O_terminal_H, E_terminal_H):
    MIN = min(O_S_B, E_S)
    MAX = max(O_S_B, E_S)
    # print("MIN, MAX = ", MIN, MAX)

    # delta upper bound from S_B

    if O_S_B * E_S == 0:
        delta_ub = 0
    elif O_S_B == E_S:
        delta_ub = (2*O_S_B - 1)
    elif O_S_B >= (E_S+2) and E_terminal_H:
        # print("min=E_S contains terminal H && can +1")
        delta_ub = (2*E_S + 1)
    elif E_S >= (O_S_B+2) and O_terminal_H:
        # print("min=O_S_B contains terminal H && can +1")
        delta_ub = (2*O_S_B + 1)
    else:
        delta_ub = 2*MIN

    # print("delta_OSB_ES = ", delta_ub)
    return delta_ub


def delta_ESB_OS(E_S_B, O_S, E_terminal_H, O_terminal_H):
    MIN = min(E_S_B, O_S)
    MAX = max(E_S_B, O_S)
    # print("MIN, MAX = ", MIN, MAX)

    # delta upper bound from S_B

    if E_S_B * O_S == 0:
        delta_ub = 0
    elif E_S_B == O_S:
        delta_ub = (2*E_S_B - 1)
    elif E_S_B >= (O_S+2) and O_terminal_H:
        # print("min=O_S contains terminal H && can +1")
        delta_ub = (2*O_S + 1)
    elif O_S >= (E_S_B+2) and E_terminal_H:
        # print("min=E_S_B contains terminal H && can +1")
        delta_ub = (2*E_S_B + 1)
    else:
        delta_ub = 2*MIN

    # print("delta_ESB_OS = ", delta_ub)
    return delta_ub


def seq_parity_stats(seq):
    Odd_H_indices = []
    Even_H_indices = []
    O_S, E_S = 0, 0

    # whether Odd or Even H is terminal end residues
    O_terminal_H = False
    E_terminal_H = False

    for index, aa in enumerate(seq):
        order = index + 1
        print(f"Order-{order}-Residue-{aa}")
        if aa == "H":
            if order % 2 == 1:
                print("H Odd")
                Odd_H_indices.append(order)
                O_S += 1
                if order == 1 or order == len(seq):
                    O_terminal_H = True
            elif order % 2 == 0:
                print("H Even")
                Even_H_indices.append(order)
                E_S += 1
                if order == len(seq):
                    E_terminal_H = True

    # convert Odd and Even H indices to np array
    Odd_H_indices = np.asarray(Odd_H_indices, dtype=int)
    Even_H_indices = np.asarray(Even_H_indices, dtype=int)
    print("Odd_H_indices = ", Odd_H_indices)
    print("Even_H_indices = ", Even_H_indices)
    print("O_S = ", O_S)
    print("E_S = ", E_S)
    print("O_terminal_H = ", O_terminal_H)
    print("E_terminal_H = ", E_terminal_H)

    # use Hart & Newman (MIT) formula Dec05 2021
    OPT_S = 2 * min(O_S, E_S) + 2
    print("OPT_S = ", OPT_S)

    return (
        Odd_H_indices,
        Even_H_indices,
        O_S,
        E_S,
        O_terminal_H,
        E_terminal_H,
        OPT_S,
    )


def early_stop_S_B(seq, step, O_S, E_S,
                    Odd_H_indices, Even_H_indices,
                    O_terminal_H, E_terminal_H, OPT_S):

    # print(f"\nstep-{step-1}")
    split_point = step + 2

    # processed seq S_A
    # remaining seq S_B
    S = seq
    S_A = S[:split_point]
    S_B = S[split_point:]
    # print(S, S_A, S_B)

    # if S_B[0] == 'P':
    #     print("ignore P residue, pass...")
    #     return OPT_S

    O_S_B = (Odd_H_indices >= (split_point+1)).sum()
    E_S_B = (Even_H_indices >= (split_point+1)).sum()
    # print("O_S_B = ", O_S_B)
    # print("E_S_B = ", E_S_B)

    # beware of backbone HHs
    if O_S_B == 1:
        for odd_H in Odd_H_indices[Odd_H_indices >= (split_point+1)]:
            # print(f"for Odd_H ({odd_H}) in O_S_B...")
            if (odd_H-1) in Even_H_indices and (odd_H+1) in Even_H_indices:
                # print('  found even-odd-even backbone_HHH')
                # print("  E_S -= 2")
                E_S -= 2
            elif (odd_H-1) in Even_H_indices:
                # print('  found odd_H-1 backbone in even-H')
                # print("  E_S -= 1")
                E_S -= 1
            elif (odd_H+1) in Even_H_indices:
                # print('  found odd_H+1 backbone in even-H')
                # print("  E_S -= 1")
                E_S -= 1
    if E_S_B == 1:
        for even_H in Even_H_indices[Even_H_indices >= (split_point+1)]:
            # print(f"for Even_H ({even_H}) in E_S_B...")
            if (even_H-1) in Odd_H_indices and (even_H+1) in Odd_H_indices:
                # print('  found odd-even-odd backbone_HHH')
                # print("  O_S -= 2")
                O_S -= 2
            elif (even_H-1) in Odd_H_indices:
                # print('  found even_H-1 backbone in odd-H')
                # print("  O_S -= 1")
                O_S -= 1
            elif (even_H+1) in Odd_H_indices:
                # print('  found even_H+1 backbone in odd-H')
                # print("  O_S -= 1")
                O_S -= 1

    # print(f"adjusted O_S: {O_S} | E_S: {E_S} | O_S_B: {O_S_B} | E_S_B: {E_S_B}")

    max_delta = delta_OSB_ES(O_S_B, E_S, O_terminal_H, E_terminal_H) + \
        delta_ESB_OS(E_S_B, O_S, E_terminal_H, O_terminal_H)
    # print("max_delta from S_B = ", max_delta)

    if max_delta >= OPT_S:
        # print("delta_S_B >= OPT_S...")
        return OPT_S
    else:
        return max_delta


def get_F_patterns(seq):
    """SAW investigate % of Fs on the energies"""
    N = len(seq)
    print("N = ", N)

    F_half_pattern = ""
    F_half_minus_one_pattern = ""

    N_half = int(np.floor(
        len(seq)/2
    ))

    # print("N_half = ", N_half)

    for _ in range(N_half):
        F_half_pattern += 'F' + ", "
    F_half_pattern = F_half_pattern[:-2]
    # print("F_half_pattern =", F_half_pattern)

    for _ in range(N_half-1):
        F_half_minus_one_pattern += 'F' + ", "
    F_half_minus_one_pattern = F_half_minus_one_pattern[:-2]
    # print("F_half_minus_one_pattern =", F_half_minus_one_pattern)

    return (
        N_half,
        F_half_pattern,
        F_half_minus_one_pattern,
    )

if __name__ == "__main__":
    # execute only if run as a script
    seq = "HPHPHHHH"

    (
        Odd_H_indices,
        Even_H_indices,
        O_S,
        E_S,
        O_terminal_H,
        E_terminal_H,
        OPT_S,
    ) = seq_parity_stats(seq)

    for step in range(15):
        max_delta = early_stop_S_B(seq, step, O_S, E_S,
                        Odd_H_indices, Even_H_indices,
                        O_terminal_H, E_terminal_H, OPT_S)
