import os
from mpi4py import MPI

os.system("python3 read_video.py")

DIR = "./frames/"
num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

MPI.Init()

comm = MPI.COMM_WORLD   # Defines the default communicator
num_procs = comm.Get_size()  # Stores the number of processes in size.

print("We have %d number of processes",num_files)

rank = comm.Get_rank()  # Stores the rank (pid) of the current process

for i in range(num_files):
    file_to_run = "python3 process_frame.py " + str(i)
    os.system(file_to_run)

MPI.Finalize()

os.system("python3 assemble_video.py")