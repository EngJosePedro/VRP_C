from pathlib import Path

def write_cvrp_instance(
    out_vrp,
    name,
    coords,   # lista [(x,y)], coords[0] = depot
    demands,  # lista [0, d1, d2, ...]
    capacity
):
    n = len(coords)
    lines = [
        f"NAME : {name}",
        "TYPE : CVRP",
        "COMMENT : Generated instance",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        #"EDGE_WEIGHT_TYPE : EXACT_2D",
        f"CAPACITY : {capacity}",
        "",
        "NODE_COORD_SECTION",
    ]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i} {int(x)} {int(y)}")

    lines.append("")
    lines.append("DEMAND_SECTION")
    for i, d in enumerate(demands, start=1):
        lines.append(f"{i} {int(d)}")

    lines.append("")
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")

    Path(out_vrp).write_text("\n".join(lines), encoding="utf-8")


def write_par_file(out_par, problem_file, output_tour_file):
    lines = [
        f"PROBLEM_FILE = {problem_file}",
        f"OUTPUT_TOUR_FILE = {output_tour_file}",
        "TRACE_LEVEL = 1",
        "RUNS = 1",
        "SEED = 1234",
        "MAX_TRIALS = 10000",
    ]
    Path(out_par).write_text("\n".join(lines), encoding="utf-8")


#coords = [
#    (50, 50),  # depot
#    (10, 10),
#    (20, 10),
#    (30, 15),
#    (60, 20),
#    (70, 30),
#    (65, 60),
#    (20, 70),
#    (10, 50),
#]

#demands = [0, 4, 3, 2, 5, 4, 3, 2, 1]
#capacity = 15

#base = Path(r"C:\Users\jose-\source\repos\LKH3")
#inst_dir = base / "instances" / "cvrp"
#sol_dir = base / "solutions" / "cvrp"
#inst_dir.mkdir(parents=True, exist_ok=True)
#sol_dir.mkdir(parents=True, exist_ok=True)

#vrp_path = inst_dir / "cvrp_8.vrp"
#par_path = inst_dir / "cvrp_8.par"
#tour_path = sol_dir / "cvrp_8.tour"

#write_cvrp_instance(vrp_path, "cvrp_8", coords, demands, capacity)
#write_par_file(par_path, vrp_path, tour_path)

#print("Arquivos gerados:")
#print(vrp_path)
#print(par_path)
