{ jsonPath } : (rec {
    # 
    # create a standard library for convienience 
    # 
    frozenStd = (builtins.import 
        (builtins.fetchTarball
            ({url="https://github.com/NixOS/nixpkgs/archive/8917ffe7232e1e9db23ec9405248fd1944d0b36f.tar.gz";})
        )
        ({})
    );
    std = (frozenStd.lib.mergeAttrs
        (builtins) # <- for import, fetchTarball, etc 
        (frozenStd.lib.mergeAttrs
            ({ stdenv = frozenStd.stdenv; })
            (frozenStd.lib) # <- for mergeAttrs, optionals, getAttrFromPath, etc 
        )
    );
    # 
    # load the nix.json cause were going to extract basically everything from there
    # 
    packageJson = (std.fromJSON
        (std.readFile
            (jsonPath)
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
    jsonPackagesWithSources = (std.map
        (each: 
            ({
                name = (std.concatMapStringsSep
                    (".")
                    (each: each)
                    (each.load)
                );
                commitHash = each.from;
                asNativeBuildInput = (
                    (std.hasAttr
                        ("asNativeBuildInput")
                        (each)
                    )
                    &&
                    each.asNativeBuildInput
                );
                value =
                    # if it says where (e.g. from)
                    if 
                        (std.hasAttr
                            ("from")
                            (each)
                        )
                    # then load it from that place
                    then 
                        (rec {
                            package = (std.getAttrFromPath
                                (each.load)
                                (std.import
                                    # if its a string, assume its a nixpkg commit hash
                                    (
                                        if 
                                            (std.isString
                                                (each.from)
                                            )
                                        then
                                            (std.fetchGit
                                                ({
                                                    url = "https://github.com/NixOS/nixpkgs";
                                                    rev = each.from;
                                                })
                                            )
                                        # otherwise assume its the details for a github repo
                                        else
                                            (std.fetchGit
                                                (each.from.fetchGit)
                                            )
                                    )
                                    (
                                        if 
                                            (std.isString
                                                (each.from)
                                            )
                                        then
                                            ({ config = packageJson.nix.config; })
                                        # otherwise assume its the details for a github repo
                                        else
                                            (each.from.options)
                                    )
                                )
                            );
                            return = (
                                if 
                                    (std.hasAttr
                                        ("override")
                                        (each)
                                    )
                                then
                                    (package.override
                                        (each.override)
                                    )
                                else
                                    package
                            );
                        }.return)
                    # otherwise just default to getting it from mainPackages
                    else 
                        (std.getAttrFromPath
                            (each.load)
                            (mainPackages)
                        )
                ;
            })
        )
        (packagesForThisMachine)
    );
    buildInputs = (std.map
        (each: each.value)
        (std.filter
            (each: !each.asNativeBuildInput)
            (jsonPackagesWithSources)
        )
    );
    nativeBuildInputs = (std.map
        (each: each.value)
        (std.filter
            (each: each.asNativeBuildInput)
            (jsonPackagesWithSources)
        )
    );
    depedencyPackages = (std.listToAttrs 
        (jsonPackagesWithSources)
    );
    return = (std.mergeAttrs
        (std)
        ({
            nixPath = "${mainRepo}";
            packages = (std.mergeAttrs
                mainPackages
                depedencyPackages
            );
            project = {
                buildInputs = buildInputs;
                nativeBuildInputs = nativeBuildInputs;
            };
            protectHomeShellCode = ''
                source "$PWD/settings/project.config.sh"
                
                if [ -n "$PROJECTR_HOME" ]
                then
                    # we don't want to give nix or other apps our home folder
                    if [[ "$HOME" != "$PROJECTR_HOME" ]] 
                    then
                        mkdir -p "$PROJECTR_HOME/.cache/"
                        ln -s "$HOME/.cache/nix" "$PROJECTR_HOME/.cache/" &>/dev/null
                        
                        # so make the home folder the same as the project folder
                        export HOME="$PROJECTR_HOME"
                        # make it explicit which nixpkgs we're using
                        export NIX_PATH="nixpkgs=${mainRepo}:."
                    fi
                fi
            '';
        })
    );
}).return