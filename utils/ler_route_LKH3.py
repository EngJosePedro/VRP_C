import numpy as np


def read_lkh_tour(filepath):
    """
    Lê o arquivo .tour do LKH e retorna a sequência de nós.
    """
    tour = []
    reading = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line == "TOUR_SECTION":
                reading = True
                continue

            if not reading:
                continue

            if line in ("-1", "EOF"):
                break

            tour.append(int(line))

    return tour


def convert_lkh_to_routes(tour, n_customers):
    """
    Converte sequência LKH em vetor com separadores de rota.

    Regras:
    - nó 1 = depósito
    - nós >= n_customers + 2 = depósitos artificiais
    - clientes: 2 até n_customers+1
    - saída: numpy array com clientes indexados em 0
    """

    result = [0]  # começa no depósito

    for node in tour:

        # depósito original ou artificial
        if node == 1 or node >= (n_customers + 2):
            # evita duplicar 0
            if result[-1] != 0:
                result.append(0)
            continue

        # cliente → converter para índice base 0
        customer = node - 2  # (2 → 0, 3 → 1, ..., n+1 → n-1)
        result.append(customer + 1)  # opcional: manter 1-based interno

    # garantir fechamento final
    if result[-1] != 0:
        result.append(0)

    return np.array(result, dtype=np.int32)