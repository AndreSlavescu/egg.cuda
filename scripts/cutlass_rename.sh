#!/bin/bash
#
# Adding "cutlass" to kernel names may trigger
# different register allocation strategies in NVCC. 

SOURCE_FILE="$1"
IS_H100="$2"

if [ "$IS_H100" != "1" ]; then
    echo "[cutlass_rename] Not H100, skipping kernel renaming"
    exit 0
fi

echo "[cutlass_rename] Applying cutlass prefix to kernels"

KERNELS=(
    "fused_perturbation_kernel"
    "gru_gate_kernel"
    "gru_ht_kernel"
    "gru_update_kernel"
    "mlp_residual_kernel"
    "loss_kernel"
    "update_weights_kernel"
)

for kernel in "${KERNELS[@]}"; do
    sed -i "s/\b${kernel}\b/cutlass_${kernel}/g" "$SOURCE_FILE"
done

echo "[cutlass_rename] Renamed ${#KERNELS[@]} kernels with cutlass_ prefix"

