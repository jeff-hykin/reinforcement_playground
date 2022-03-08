# 
# how to add packages?
# 
    # you can search for them here: https://search.nixos.org/packages
    # to find them in the commandline use:
    #     nix-env -qP --available PACKAGE_NAME_HERE | cat
    # ex:
    #     nix-env -qP --available opencv
    #
    # NOTE: some things (like setuptools) just don't show up in the 
    # search results for some reason, and you just have to guess and check 🙃 

{
    # 
    # load most things from the nix.toml
    # 
    main ? (builtins.import
        (builtins.getEnv
            ("__FORNIX_NIX_MAIN_CODE_PATH")
        )
    ),
    nixgl ? (builtins.import
        (fetchTarball "https://github.com/guibou/nixGL/archive/17658df1e17a64bc23ee5c93cfa9e8b663a4ac81.tar.gz")
        {}
    )
}:
    # Lets setup some definitions
    let        
        # just a helper
        emptyOptions = ({
            buildInputs = [];
            nativeBuildInputs = [];
            shellCode = "";
        });
        
        salt = (builtins.import
            (./nixpkgs/salt.nix)
            main
        );
        
        # torch = (builtins.import
        #     (fetchTarball "https://github.com/jeff-hykin/pytorch_nixpkg/archive/77961dce25528445a7e4e448652754079deb6f73.tar.gz")
        #     {
        #         pkgs = main.packages // {
        #             cudnn_cudatoolkit_11_2 = main.packages.cudnn_cudatoolkit_11_2;
        #         };
        #     }
        # );
        magma = (main.packages.magma.override
            ({
                cudatoolkit = main.packages.cudaPackages.cudatoolkit_11_2;
            })
        );
        nccl_cudatoolkit_11_2 = (builtins.import
            (builtins.fetchTarball 
                ({
                    url = "https://github.com/NixOS/nixpkgs/archive/2cdd608fab0af07647da29634627a42852a8c97f.tar.gz";
                })
            )
            ({})
        ).nccl_cudatoolkit_11;

        
        # 
        # Linux Only
        #
        linuxOnly = if main.stdenv.isLinux then ({
            buildInputs = [
                nixgl.auto.nixGLNvidia
                main.packages.cudaPackages.cudatoolkit_11_2
                main.packages.cudnn_cudatoolkit_11_2
                (main.packages.python38Packages.pytorchWithCuda.override 
                    ({
                        cudaSupport = true;
                        cudatoolkit = main.packages.cudaPackages.cudatoolkit_11_2;
                        cudnn = main.packages.cudnn_cudatoolkit_11_2;
                        nccl = nccl_cudatoolkit_11_2;
                        magma = magma;
                    })
                )
                # (torch { name = "torch";})
            ];
            nativeBuildInputs = [];
            shellCode = ''
                if [[ "$OSTYPE" == "linux-gnu" ]] 
                then
                    true # add important (LD_LIBRARY_PATH, PATH, etc) nix-Linux code here
                    export CUDA_PATH="${main.packages.cudaPackages.cudatoolkit_11_2}"
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${main.packages.linuxPackages.nvidia_x11}/lib:${main.packages.ncurses5}/lib:/run/opengl-driver/lib"
                    export EXTRA_LDFLAGS="$EXTRA_CCFLAGS:-L/lib -L${main.packages.linuxPackages.nvidia_x11}/lib"
                    export LD_LIBRARY_PATH="$(${nixgl.auto.nixGLNvidia}/bin/nixGLNvidia-470.86 printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
                    export EXTRA_CCFLAGS="$EXTRA_CCFLAGS:-I/usr/include"
                    export LD_LIBRARY_PATH="${main.makeLibraryPath [ main.packages.glib ] }:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${main.packages.ncurses}/lib:/run/opengl-driver/lib"
                    
                    export LD_LIBRARY_PATH="${main.packages.hdf5}:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${main.packages.openmpi}/lib:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${main.packages.python38}/lib:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${main.packages.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                    export LD_LIBRARY_PATH="${main.packages.zlib}/lib:$LD_LIBRARY_PATH"

                    # CUDA and magma path
                    export LD_LIBRARY_PATH="${main.packages.cudaPackages.cudatoolkit_11_2}/lib:${main.packages.cudnn_cudatoolkit_11_2}/lib:${magma}/lib:$LD_LIBRARY_PATH"
                fi
            '';
            # for python with CUDA 
            # 1. install cuda drivers on the main machine then
            # 2. include the following inside the shellCode if statement above
            #     export CUDA_PATH="${main.packages.cudatoolkit}"
            #     export EXTRA_LDFLAGS="-L/lib -L${main.packages.linuxPackages.nvidia_x11}/lib"
            #     export EXTRA_CCFLAGS="-I/usr/include"
            #     export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${main.packages.linuxPackages.nvidia_x11}/lib:${main.packages.ncurses5}/lib:/run/opengl-driver/lib"
            #     export LD_LIBRARY_PATH="$(${main.packages.nixGLNvidia}/bin/nixGLNvidia printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
            #     export LD_LIBRARY_PATH="${main.makeLibraryPath [ main.packages.glib ] }:$LD_LIBRARY_PATH"
            # 3. then add the following to the nix.toml file
            #    # 
            #    # Nvidia
            #    # 
            #    [[packages]]
            #    load = [ "nixGLNvidia",]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #    # see https://discourse.nixos.org/t/opencv-with-cuda-in-nix-shell/7358/5
            #    from = { fetchGit = { url = "https://github.com/guibou/nixGL", rev = "7d6bc1b21316bab6cf4a6520c2639a11c25a220e" }, }
            # 
            #    [[packages]]
            #    load = [ "pkgconfig",]
            #    asNativeBuildInput = true
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            # 
            #    [[packages]]
            #    load = [ "cudatoolkit",]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            #    [[packages]]
            #    load = [ "libconfig",]
            #    asNativeBuildInput = true
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            #    [[packages]]
            #    load = [ "cmake",]
            #    asNativeBuildInput = true
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            #    [[packages]]
            #    load = [ "libGLU",]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            #    [[packages]]
            #    load = [ "linuxPackages", "nvidia_x11",]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            #    [[packages]]
            #    load = [ "stdenv", "cc",]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #
            # 4. if you want opencv with cuda add the following to the nix.toml
            #    # 
            #    # opencv
            #    # 
            #    [[packages]]
            #    onlyIf = [ [ "stdenv", "isLinux",],]
            #    load = [ "opencv4",]
            #    override = { enableGtk3 = true, enableFfmpeg = true, enableCuda = true, enableUnfree = true, }
            #    # see https://discourse.nixos.org/t/opencv-with-cuda-in-nix-shell/7358/5
            #    from = { fetchGit = { url = "https://github.com/NixOS/nixpkgs/", rev = "a332da8588aeea4feb9359d23f58d95520899e3c" }, options = { config = { allowUnfree = true } }, }
        }) else emptyOptions;
        
        # 
        # Mac Only
        # 
        macOnly = if main.stdenv.isDarwin then ({
            buildInputs = [];
            nativeBuildInputs = [];
            shellCode = ''
                if [[ "$OSTYPE" = "darwin"* ]] 
                then
                    true # add important nix-MacOS code here
                    export EXTRA_CCFLAGS="$EXTRA_CCFLAGS:-I/usr/include:${main.packages.darwin.apple_sdk.frameworks.CoreServices}/Library/Frameworks/CoreServices.framework/Headers/"
                fi
            '';
        }) else emptyOptions;
        
    # using the above definitions
    in
        # 
        # create a shell
        # 
        main.packages.mkShell {
            # inside that shell, make sure to use these packages
            buildInputs =  main.project.buildInputs ++ macOnly.buildInputs ++ linuxOnly.buildInputs ++ [ salt ];
            
            nativeBuildInputs =  main.project.nativeBuildInputs ++ macOnly.nativeBuildInputs ++ linuxOnly.nativeBuildInputs;
            
            # run some bash code before starting up the shell
            shellHook = ''
                ${main.project.protectHomeShellCode}
                if [ "$FORNIX_DEBUG" = "true" ]; then
                    echo "starting: 'shellHook' inside the 'settings/extensions/nix/shell.nix' file"
                fi
                ${linuxOnly.shellCode}
                ${macOnly.shellCode}
                
                # provide access to ncurses for nice terminal interactions
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${main.packages.ncurses}/lib"
                export LD_LIBRARY_PATH="${main.makeLibraryPath [ main.packages.glib ] }:$LD_LIBRARY_PATH"
                
                if [ "$FORNIX_DEBUG" = "true" ]; then
                    echo "finished: 'shellHook' inside the 'settings/extensions/nix/shell.nix' file"
                    echo ""
                    echo "Tools/Commands mentioned in 'settings/extensions/nix/nix.toml' are now available/installed"
                fi
            '';
        }
