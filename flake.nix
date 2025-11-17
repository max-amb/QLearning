{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        with pkgs;
        {
          devShells.default = mkShell {
            buildInputs = [ clang cmake eigen_3_4_0 pkg-config  python313Packages.pybind11 python313 ];
                    shellHook = ''
            export PYTHON_EXECUTABLE="${python313}/bin/python3"
            export Python3_ROOT_DIR="${python313}"
          '';
          };
        }
      );
}
