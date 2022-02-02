this_folder="$FORNIX_FOLDER/main/world_builders/carla"
lib_path="$this_folder/lib"
file_path="$lib_path/libomp.so.5"
# check if file exists, and if not
if ! [ -f "$file_path" ]
then
    openmp_path="$("$FORNIX_FOLDER/commands/tools/nix/package_path_for" llvmPackages.openmp)"
    if [ -n "$openmp_path" ]
    then
        # create the target folder
        mkdir -p "$lib_path"
        # create the file
        ln -s "$openmp_path/lib/libomp.so" "$file_path"
    fi
fi
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$lib_path"

# SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 workspace.ignore/CarlaUE4.sh -ResX=800 -ResY=600 -nosound -headless -opengl