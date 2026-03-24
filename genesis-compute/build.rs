fn main() {
    #[cfg(feature = "cpp")]
    {
        cc::Build::new()
            .cpp(true)
            .flag("-fopenmp")
            .flag("-O3")
            .flag("-march=native")
            .file("kernels/cpu_kernels.cpp")
            .compile("genesis_kernels");
        println!("cargo:rustc-link-lib=static=genesis_kernels");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rerun-if-changed=kernels/cpu_kernels.cpp");
    }

    // CUDA build skeleton (uncomment when nvcc is available)
    /*
    if std::env::var("CUDA_HOME").is_ok() {
        cc::Build::new()
            .cuda(true)
            .file("kernels/cuda_kernels.cu")
            .compile("genesis_cuda");
    }
    */
}
