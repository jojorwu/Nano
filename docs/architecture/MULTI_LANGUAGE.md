# Multi-Language Architecture (Nano v4.2)

## Overview
Nano v4.2 introduces a modular, multi-language architecture to balance performance, flexibility, and ease of use.

## Languages & Roles

| Language | Component | Role |
| :--- | :--- | :--- |
| **Rust** | `genesis-core`, `genesis-node` | **Core Orchestrator:** Memory safety, thread management, and FFI glue. |
| **C++** | `genesis-compute/kernels` | **High-Performance CPU Kernels:** SIMD-optimized spike propagation and state updates. |
| **CUDA** | `genesis-compute/kernels` | **GPU Acceleration:** Massively parallel simulation on NVIDIA hardware. |
| **Python** | `genesis-client`, `genesis-python` | **SDK & Research:** Rapid prototyping, model building, and high-level control. |

## Data Flow (Zero-Copy FFI)
Data is stored in **Structure of Arrays (SoA)** format in Rust.
FFI-safe pointers (`NeuronsFFI`) allow C++ and Python (via PyO3) to access this memory directly without copying.

## Build System
- **Cargo (Rust):** Primary build tool.
- **CC Crate:** Compiles C++ and CUDA kernels during the Rust build process.
- **PyO3:** Generates Python extension modules.
