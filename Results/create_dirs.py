import os

i_values = [0, 1]
j_values = range(7)
k_values = range(7)
p_values = range(10)
t_values = range(5)

j_size = [32, 64, 128, 256, 512, 1024, 2048]
k_size = [100, 200, 500, 1000, 2000, 5000, 10000]

def create_bash_file(directory, command):
    with open(os.path.join(directory, "run_script.sh"), "w") as file:
        file.write(command)

for p in p_values:
    gpu_directory = f"gpu_solv{p}"
    if not os.path.exists(gpu_directory):
        os.makedirs(gpu_directory)
        
    for j in j_values:
        for k in k_values:
            if p == 4:
                for t in t_values:
                    t_directory_name = f"gpu_solv{p}_{j}_{k}_{t}"
                    t_full_path = os.path.join(gpu_directory, t_directory_name)
                    if not os.path.exists(t_full_path):
                        os.makedirs(t_full_path)
                    command = f"python ../../../gpu_heat.py -m {j_size[j]} {j_size[j]} {j_size[j]} -N {k_size[k]} -s {p} -p {t} >> times.txt"
                    create_bash_file(t_full_path, command)
            
            else:

                directory_name = f"gpu_solv{p}_{j}_{k}"
                full_path = os.path.join(gpu_directory, directory_name)

                if not os.path.exists(full_path):
                    os.makedirs(full_path)

                command = f"python ../../../gpu_heat.py -m {j_size[j]} {j_size[j]} {j_size[j]} -N {k_size[k]} -s {p} >> times.txt"
                create_bash_file(full_path, command)

for i in i_values:
    for j in j_values:
        for k in k_values:
            seq_directory = f"seq_solv{i}"
            if not os.path.exists(seq_directory):
                os.makedirs(seq_directory)
            
            directory_name = f"seq_solv{i}_{j}_{k}"
            full_path = os.path.join(seq_directory, directory_name)

            if not os.path.exists(full_path):
                os.makedirs(full_path)
            
            command = f"python ../../../sequential_heat.py -m {j_size[j]} {j_size[j]} {j_size[j]} -N {k_size[k]} -s {i} >> times.txt"
            create_bash_file(full_path, command)