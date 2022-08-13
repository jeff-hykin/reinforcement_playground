{ main }:
    let
        # 
        # sources
        # 
        nixgl = (builtins.import
            (fetchTarball "https://github.com/guibou/nixGL/archive/047a34b2f087e2e3f93d43df8e67ada40bf70e5c.tar.gz")
            {}
        );
        pkgsWithTorch_1_8_1 = (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/141439f6f11537ee349a58aaf97a5a5fc072365c.tar.gz";
                })
            )
            ({})
        );
        pkgsWithTorch_1_9_0 = (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/c82b46413401efa740a0b994f52e9903a4f6dcd5.tar.gz";
                })
            )
            ({})
        );
        pkgsWithTorch_1_9_1 = (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/5e15d5da4abb74f0dd76967044735c70e94c5af1.tar.gz";
                })
            )
            ({})
        );
        pkgsWithNcclCudaToolkit_11_2 =  (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/2cdd608fab0af07647da29634627a42852a8c97f.tar.gz";
                })
            )
            ({})
        );
        pkgsWithWorkingShap =  (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/3f50332bc4913a56ad216ca01f5d0bd24277a6b2.tar.gz";
                })
            )
            ({})
        );
        
        # 
        # 
        # packages
        # 
        # 
        python = pkgsWithTorch_1_9_0.python38;
        pythonPackages = pkgsWithTorch_1_9_0.python38Packages;
        selectedCommonPythonPackages = [
            pythonPackages.black
            pythonPackages.poetry
            pythonPackages.setuptools
            pythonPackages.pyopengl
            pythonPackages.pip
            pythonPackages.virtualenv
            pythonPackages.wheel
            pythonPackages.numpy
            pkgsWithWorkingShap.python38Packages.shap
        ];
        
        # 
        # for pytorch
        # 
        cudaStuff = (
            let 
                cudatoolkit = pkgsWithTorch_1_9_0.cudaPackages.cudatoolkit_11_2;
                cudnn = pkgsWithTorch_1_9_0.cudnn_cudatoolkit_11_2;
                nccl = pkgsWithNcclCudaToolkit_11_2.nccl_cudatoolkit_11;
                magma = (pkgsWithTorch_1_9_0.magma.override
                    ({
                        cudatoolkit = cudatoolkit;
                    })
                );
                pytorchWithCuda = (pythonPackages.pytorchWithCuda.override 
                    ({
                        cudaSupport = true;
                        cudatoolkit = cudatoolkit;
                        cudnn = cudnn;
                        nccl = nccl;
                        magma = magma;
                    })
                );
                nvidia_x11 = pkgsWithTorch_1_9_0.linuxPackages.nvidia_x11;
            in
                # return all this stuff
                {
                    cudatoolkit = cudatoolkit;
                    cudnn = cudnn;
                    magma = magma;
                    nccl = nccl;
                    pytorchWithCuda = pytorchWithCuda;
                    nvidia_x11 = nvidia_x11;
                    pythonPackages = pkgsWithTorch_1_9_0.python38Packages;
                }
        );
        shouldEnableCuda = (main.getEnv "__ENABLE_CUDA_FOR_FORNIX") == "true";
    in
        if !shouldEnableCuda
        then
            {
                buildInputs = [
                    python
                ];
            }
        else
            {
                buildInputs = selectedCommonPythonPackages ++ [
                    pythonPackages.pybullet
                    nixgl.auto.nixGLNvidia
                    cudaStuff.cudatoolkit
                    cudaStuff.cudnn
                    cudaStuff.pytorchWithCuda
                ];
                nativeBuildInputs = [];
                shellHook = ''
                    if [[ "$OSTYPE" == "linux-gnu" ]] 
                    then
                        true # add important (LD_LIBRARY_PATH, PATH, etc) nix-Linux code here
                        export CUDA_PATH="${cudaStuff.cudatoolkit}"
                        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${cudaStuff.nvidia_x11}/lib"
                        export EXTRA_LDFLAGS="$EXTRA_CCFLAGS:-L/lib -L${cudaStuff.nvidia_x11}/lib"
                        export LD_LIBRARY_PATH="$(${nixgl.auto.nixGLNvidia}/bin/nixGLNvidia-470.86 printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
                        export EXTRA_CCFLAGS="$EXTRA_CCFLAGS:-I/usr/include"
                        export LD_LIBRARY_PATH="${main.makeLibraryPath [ main.packages.glib ] }:$LD_LIBRARY_PATH"
                        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
                        
                        export LD_LIBRARY_PATH="${main.packages.hdf5}:$LD_LIBRARY_PATH"
                        export LD_LIBRARY_PATH="${main.packages.openmpi}/lib:$LD_LIBRARY_PATH"
                        export LD_LIBRARY_PATH="${main.packages.python38}/lib:$LD_LIBRARY_PATH"
                        export LD_LIBRARY_PATH="${main.packages.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                        export LD_LIBRARY_PATH="${main.packages.zlib}/lib:$LD_LIBRARY_PATH"

                        # CUDA and magma path
                        export LD_LIBRARY_PATH="${cudaStuff.cudatoolkit}/lib:${cudaStuff.cudnn}/lib:${cudaStuff.magma}/lib:$LD_LIBRARY_PATH"
                    fi
                '';
            }