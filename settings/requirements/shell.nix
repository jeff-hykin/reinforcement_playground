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
    # 
    # 
    # nix.json
    # 
    # 
        frozenStd = (builtins.import 
            (builtins.fetchTarball
                ({url="https://github.com/NixOS/nixpkgs/archive/a332da8588aeea4feb9359d23f58d95520899e3c.tar.gz";})
            )
            ({})
        ).lib;
        std = (frozenStd.mergeAttrs
            (builtins) # <- for import, fetchTarball, etc 
            (frozenStd) # <- for mergeAttrs, optionals, getAttrFromPath, etc 
        );
        # load packages and config
        definitions = rec {
            # 
            # load the nix.json cause were going to extract basically everything from there
            # 
            packageJson = (std.fromJSON
                (std.readFile
                    (./nix.json)
                )
            );
            # 
            # load the store with all the packages, and load it with the config
            # 
            mainRepo = (std.fetchTarball
                ({url="https://github.com/NixOS/nixpkgs/archive/${packageJson.nix.mainRepo}.tar.gz";})
            );
            mainPackages = (std.import
                (mainRepo)
                ({ config = packageJson.nix.config;})
            );
            packagesForThisMachine = (std.filter
                (eachPackage:
                    (std.all
                        # if all are true
                        (x: x)
                        (std.optionals
                            # if package depends on something
                            (std.hasAttr "onlyIf" eachPackage)
                            # basically convert something like ["stdev", "isLinux"] to std.stdenv.isLinux
                            (std.map
                                (eachCondition:
                                    
                                    (std.getAttrFromPath
                                        (eachCondition)
                                        (std)
                                    )
                                )
                                (eachPackage.onlyIf)
                            )
                        )
                    )
                )
                (packageJson.nix.packages)
            );
            # 
            # reorganize the list of packages from:
            #    [ { load: "blah", from:"blah-hash" }, ... ]
            # into a list like:
            #    [ { name: "blah", commitHash:"blah-hash", source: (*an object*) }, ... ]
            #
            packagesWithSources = (std.map
                (each: 
                    ({
                        name = (std.concatMapStringsSep
                            (".")
                            (each: each)
                            (each.load)
                        );
                        commitHash = each.from;
                        source = if std.isString
                            then (std.getAttrFromPath
                                (each.load)
                                (std.import 
                                    (std.fetchTarball
                                        ({url="https://github.com/NixOS/nixpkgs/archive/${each.from}.tar.gz";})
                                    ) 
                                    ({ config = packageJson.nix.config;})
                                )
                            )
                            else (mainPackages)
                        ;
                    })
                )
                (packagesForThisMachine)
            );
        };
    # 
    # 
    # Conditional Dependencies
    # 
    # 
        # TODO: add support for the nix.json to have OS specific sections so this is no longer needed
        
        # 
        # Linux Only
        # 
        linuxOnlyPackages = [] ++ std.optionals (definitions.mainPackages.stdenv.isLinux) [
            definitions.mainPackages.stdenv.cc
            definitions.mainPackages.linuxPackages.nvidia_x11
            definitions.mainPackages.cudatoolkit
            definitions.mainPackages.libGLU
            majorCustomDependencies.nixGL
            # opencv4cuda, see https://discourse.nixos.org/t/opencv-with-cuda-in-nix-shell/7358/5
            (definitions.mainPackages.opencv4.override {  
                enableGtk3   = true; 
                enableFfmpeg = true; 
                enableCuda   = true;
                enableUnfree = true; 
            })
        ];
        linuxOnlyNativePackages = [] ++ std.optionals (definitions.mainPackages.stdenv.isLinux) [
            definitions.mainPackages.pkgconfig
            definitions.mainPackages.libconfig
            definitions.mainPackages.cmake
        ];
        linuxOnlyShellCode = if !definitions.mainPackages.stdenv.isLinux then "" else ''
            if [[ "$OSTYPE" == "linux-gnu" ]] 
            then
                export CUDA_PATH="${definitions.mainPackages.cudatoolkit}"
                export EXTRA_LDFLAGS="-L/lib -L${definitions.mainPackages.linuxPackages.nvidia_x11}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${definitions.mainPackages.linuxPackages.nvidia_x11}/lib:${definitions.mainPackages.ncurses5}/lib:/run/opengl-driver/lib"
                export LD_LIBRARY_PATH="$(${majorCustomDependencies.nixGL}/bin/nixGLNvidia printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="${std.makeLibraryPath [ definitions.mainPackages.glib ] }:$LD_LIBRARY_PATH"
            fi
        '';
        
        # 
        # Mac Only
        # 
        macOnlyPackages = [] ++ std.optionals (definitions.mainPackages.stdenv.isDarwin) [
        ];
        macOnlyNativePackages = [] ++ std.optionals (definitions.mainPackages.stdenv.isDarwin) [
        ];
        macOnlyShellCode = if !definitions.mainPackages.stdenv.isDarwin then "" else ''
        '';
        
    # 
    # 
    # Complex Depedencies
    # 
    # 
        majorCustomDependencies = rec {
            packagesFrom_2020_11_5 = import (std.fetchGit {
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
            nixGL = (std.callPackage (
                    std.fetchGit {
                    url = "https://github.com/guibou/nixGL";
                    rev = "7d6bc1b21316bab6cf4a6520c2639a11c25a220e";
                }
            ) {}).nixGLNvidia;
        };
        
        subDepedencies = [] ++ majorCustomDependencies.python;
    
# using those definitions
in
    # create a shell
    std.mkShell {
        # inside that shell, make sure to use these packages
        buildInputs = subDepedencies ++ macOnlyPackages ++ linuxOnlyPackages ++ std.map (each: each.source) definitions.packagesWithSources;
        
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
            export NIX_PATH="nixpkgs=${definitions.mainRepo}:."
        fi
        '';
    }
