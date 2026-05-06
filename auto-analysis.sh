mpirun -np 1 python TaskKarmanVortexStreet.py "serial"

ranks=(1 2 4 8 16 32 64)

for np in "${ranks[@]}"; do
    echo "Running with $np ranks"
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    mpirun -np "$np" TaskKarmanVortexStreet.py "parallel_main"
done
