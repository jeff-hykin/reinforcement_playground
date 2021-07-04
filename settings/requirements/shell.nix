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

# Lets setup some definitions
let        
    main = (
        (builtins.import
            (../nix/parse_json_dependencies.nix)
        )
        
        ({
            jsonPath = (./nix.json);
        })
    );
    
    # 
    # 
    # Conditional Dependencies
    # 
    # 
        # TODO: add support for the nix.json to have OS specific sections so this is no longer needed
        
        # 
        # Linux Only
        # 
        linuxOnlyPackages = [] ++ main.optionals (main.packages.stdenv.isLinux) [
            main.packages.stdenv.cc
            main.packages.linuxPackages.nvidia_x11
            main.packages.cudatoolkit
            main.packages.libGLU
            majorCustomDependencies.nixGL
            # opencv4cuda, see https://discourse.nixos.org/t/opencv-with-cuda-in-nix-shell/7358/5
            (main.packages.opencv4.override {  
                enableGtk3   = true; 
                enableFfmpeg = true; 
                enableCuda   = true;
                enableUnfree = true; 
            })
        ];
        linuxOnlyNativePackages = [] ++ main.optionals (main.packages.stdenv.isLinux) [
            main.packages.pkgconfig
            main.packages.libconfig
            main.packages.cmake
        ];
        linuxOnlyShellCode = if !main.packages.stdenv.isLinux then "" else ''
            if [[ "$OSTYPE" == "linux-gnu" ]] 
            then
                export CUDA_PATH="${main.packages.cudatoolkit}"
                export EXTRA_LDFLAGS="-L/lib -L${main.packages.linuxPackages.nvidia_x11}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${main.packages.linuxPackages.nvidia_x11}/lib:${main.packages.ncurses5}/lib:/run/opengl-driver/lib"
                export LD_LIBRARY_PATH="$(${majorCustomDependencies.nixGL}/bin/nixGLNvidia printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="${main.makeLibraryPath [ main.packages.glib ] }:$LD_LIBRARY_PATH"
            fi
        '';
        
        # 
        # Mac Only
        # 
        macOnlyPackages = [] ++ main.optionals (main.packages.stdenv.isDarwin) [
        ];
        macOnlyNativePackages = [] ++ main.optionals (main.packages.stdenv.isDarwin) [
        ];
        macOnlyShellCode = if !main.packages.stdenv.isDarwin then "" else ''
        '';
        
    # 
    # 
    # Complex Depedencies
    # 
    # 
        majorCustomDependencies = rec {
            packagesFrom_2020_11_5 = import (main.fetchGit {
                # Descriptive name to make the store path easier to identify                
                name = "my-old-revision";                                                 
                url = "https://github.com/NixOS/nixpkgs/";                       
                ref = "refs/heads/nixpkgs-unstable";                     
                rev = "3f50332bc4913a56ad216ca01f5d0bd24277a6b2";
            }) {};

            python = [
                packagesFrom_2020_11_5.poetry
                packagesFrom_2020_11_5.python38
                packagesFrom_2020_11_5.python38Packages.setuptools
                packagesFrom_2020_11_5.python38Packages.pip
                packagesFrom_2020_11_5.python38Packages.virtualenv
                packagesFrom_2020_11_5.python38Packages.wheel
                packagesFrom_2020_11_5.python38Packages.shap
            ];
            
            # nixGLNvidia, see https://discourse.nixos.org/t/opencv-with-cuda-in-nix-shell/7358/5
            nixGL = (main.callPackage (
                    main.fetchGit {
                    url = "https://github.com/guibou/nixGL";
                    rev = "7d6bc1b21316bab6cf4a6520c2639a11c25a220e";
                }
            ) {}).nixGLNvidia;
        };
        
        subDepedencies = [] ++ majorCustomDependencies.python;
    
# using those definitions
in
    # create a shell
    main.packages.mkShell {
        # inside that shell, make sure to use these packages
        buildInputs = subDepedencies ++ macOnlyPackages ++ linuxOnlyPackages ++ main.project.buildInputs;
        
        nativeBuildInputs = [] ++ linuxOnlyNativePackages ++ macOnlyNativePackages;
        
        # run some bash code before starting up the shell
        shellHook = ''
        
        ${linuxOnlyShellCode}
        ${macOnlyShellCode}
        
        source "$PWD/settings/project.config.sh"
        
        # we don't want to give nix or other apps our home folder
        if [[ "$HOME" != "$PROJECTR_HOME" ]] 
        then
            mkdir -p "$PROJECTR_HOME/.cache/"
            ln -s "$HOME/.cache/nix" "$PROJECTR_HOME/.cache/" &>/dev/null
            
            # so make the home folder the same as the project folder
            export HOME="$PROJECTR_HOME"
            # make it explicit which nixpkgs we're using
            export NIX_PATH="nixpkgs=${main.nixPath}:."
        fi
        '';
    }
