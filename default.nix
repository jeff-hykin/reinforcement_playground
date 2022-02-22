{
    pkgs ? import <nixpkgs> {},
    nixgl ? import (fetchTarball "https://github.com/guibou/nixGL/archive/main.tar.gz") {}
}:

pkgs.mkShell {
  name = "cuda-sdl2-env-shell";
  buildInputs = [
    nixgl.auto.nixGLNvidia
    pkgs.cudaPackages.cudatoolkit_11_1
    pkgs.zsh
    # pkgs.SDL2
    # pkgs.SDL2_ttf
  ];
  shellHook = ''
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit_11_1}
    export EXTRA_LDFLAGS="-L/lib"
    export EXTRA_CCFLAGS="-I/usr/include/"
  '';

}
