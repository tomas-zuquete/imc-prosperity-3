import math


def find_equilibrium_realistic(multipliers,
                               N=10000,
                               iterations=2000,
                               step_size=0.0005,
                               tol=1e-12):
    """
    Attempt to find a stable distribution p[s] over 'pick sets' for crates,
    modeling that each crate i is split among n_i*N people if fraction n_i chooses it.

    multipliers[i]: multiplier for crate i
    N: total number of players
    iterations: max number of mass-shifting steps
    step_size: how much probability to move each iteration
    tol: convergence tolerance

    Returns
    -------
    p : list of floats
        final distribution p[s] (summing to 1),
        for s in set_list
    set_list : list of tuples
        each element is a tuple representing which crates form that set
    """

    # Build sets: all singletons + all pairs
    M = len(multipliers)
    V = [m * 10000 for m in multipliers]  # pot for each crate
    set_list = []
    # singletons
    for i in range(M):
        set_list.append((i,))
    # pairs
    for i in range(M):
        for j in range(i + 1, M):
            set_list.append((i, j))
    K = len(set_list)

    # Initialize p[s] = uniform
    p = [1.0 / K] * K

    def compute_fractions(p):
        """Compute n[i] = fraction choosing crate i."""
        n = [0.0] * M
        for s_idx, s in enumerate(set_list):
            for i in s:
                n[i] += p[s_idx]
        return n

    def compute_payoffs(p, n):
        """
        payoff[s] = sum_{i in s} [V_i / (n_i*N)] in an ideal sense;
        but we can ignore dividing by N for ranking and do payoff[s] = sum_{i in s} [V_i / n_i].
        If n_i == 0, treat that as huge payoff (since you'd basically get the entire pot alone).
        """
        payoffs = []
        for s in set_list:
            val = 0.0
            for i in s:
                if n[i] < 1e-12:
                    # nobody there => if you alone join, you'd share with ~0
                    val += V[i] / 1e-9  # artificially large
                else:
                    val += V[i] / n[i]
            payoffs.append(val)
        return payoffs

    for it in range(iterations):
        n = compute_fractions(p)
        payoffs = compute_payoffs(p, n)
        max_pay = max(payoffs)

        # Find all sets that achieve the top payoff
        best_sets = [idx for idx, val in enumerate(payoffs)
                     if abs(val - max_pay) < 1e-12]

        # Shift a small fraction of probability from lower-payoff sets to best sets
        delta = [0.0] * K
        changed = False

        for s_idx in range(K):
            if p[s_idx] > 1e-15:
                if payoffs[s_idx] + 1e-12 < max_pay:
                    amt = step_size * p[s_idx]
                    delta[s_idx] -= amt
                    for b_idx in best_sets:
                        delta[b_idx] += amt / len(best_sets)
                    changed = True

        new_p = [p[i] + delta[i] for i in range(K)]

        # re-normalize in case small rounding
        tot = sum(new_p)
        if tot < 1e-15:
            # degenerate => fallback to uniform
            new_p = [1.0 / K] * K
            tot = 1.0
        else:
            new_p = [x / tot for x in new_p]

        # check distance
        dist = sum(abs(new_p[i] - p[i]) for i in range(K))
        p = new_p

        if dist < tol and changed == False:
            break

    return p, set_list


def best_two_crates(p, set_list, multipliers, N, container_fees=None):
    """
    Given the equilibrium distribution p[s], compute your best pair of crates.

    Steps:
    1) Compute n[i], fraction of population in crate i.
    2) For each pair (i,j),
         your share = V[i]/(n[i]*N + 1) + V[j]/(n[j]*N + 1)
         minus container fees if any.
    3) Return the best (i,j) and the resulting payoff.
    """
    import math

    if container_fees is None:
        # default to zero cost
        container_fees = [0] * len(multipliers)

    M = len(multipliers)
    V = [m * 10000 for m in multipliers]

    # 1) compute n[i]
    n = [0.0] * M
    for s_idx, s in enumerate(set_list):
        for i in s:
            n[i] += p[s_idx]

    best_pair = None
    best_payoff = -1e15

    # 2) check all pairs i<j
    for i in range(M):
        for j in range(i + 1, M):
            # your share if you join i & j
            share_i = V[i] / ((n[i] * N) + 1)
            share_j = V[j] / ((n[j] * N) + 1)
            # subtract fees
            net_payoff = share_i + share_j - (container_fees[i] + container_fees[j])
            if net_payoff > best_payoff:
                best_payoff = net_payoff
                best_pair = (i, j)

    return best_pair, best_payoff


if __name__ == "__main__":
    multipliers = [10, 37, 17, 31, 90, 50, 20, 89, 80, 73]
    N = 10000

    # Let's say each crate has a fixed fee of 5 SeaShells, for demonstration
    container_fees = [50] * len(multipliers)

    # 1) Find equilibrium
    p_final, set_list = find_equilibrium_realistic(multipliers, N=N, iterations=20000)

    # distribution (only non-trivial probabilities)
    print("Final distribution (approx equilibrium):")
    for s_idx, s in enumerate(set_list):
        if p_final[s_idx] > 0.001:
            print(f"  Set {s} -> p = {p_final[s_idx]:.4f}")

    # 2) Determine your personal best pair
    best_pair, best_payoff = best_two_crates(p_final,
                                                     set_list,
                                                     multipliers,
                                                     N,
                                                     container_fees)

    print("\nYour best pair:", best_pair)
    print("Best pair payoff (after fees):", f"{best_payoff:.2f}")


