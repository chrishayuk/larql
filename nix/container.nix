#
# nix/container.nix - LARQL OCI container images
#
# Uses buildLayeredImage for better Docker layer caching.
# Container runs as non-root 'larql' user (UID 1000).
#
# Build:
#   nix build .#container          # larql-server image
#   nix build .#container-cli      # larql CLI image
#
# Load & Run:
#   docker load < result
#   docker run -d -p 8080:8080 -v /path/to/vindexes:/data larql-server:latest /data/my.vindex
#
# With gRPC:
#   docker run -d -p 8080:8080 -p 50051:50051 \
#     -v /path/to/vindexes:/data larql-server:latest \
#     /data/my.vindex --grpc-port 50051
#
{ pkgs, lib, larql }:
let
  user = {
    name = "larql";
    uid = 1000;
    gid = 1000;
  };

  # NSS files for user/group resolution (required for non-root containers)
  passwd = pkgs.writeTextDir "etc/passwd" ''
    root:x:0:0:root:/root:/bin/sh
    ${user.name}:x:${toString user.uid}:${toString user.gid}:larql:/home/${user.name}:/bin/sh
  '';

  group = pkgs.writeTextDir "etc/group" ''
    root:x:0:
    ${user.name}:x:${toString user.gid}:
  '';

  # Shared base contents for both images
  baseContents = [
    pkgs.bashInteractive
    pkgs.coreutils
    pkgs.cacert        # TLS certificates for HuggingFace downloads
    passwd
    group
  ];

  # ─── larql-server Container ────────────────────────────────────���─────────
  server = pkgs.dockerTools.buildLayeredImage {
    name = "larql-server";
    tag = "latest";

    contents = baseContents ++ [ larql ];

    extraCommands = ''
      mkdir -p home/${user.name}
      mkdir -p data
      mkdir -p tmp
      chmod 1777 tmp
    '';

    fakeRootCommands = ''
      chown -R ${toString user.uid}:${toString user.gid} home/${user.name}
      chown -R ${toString user.uid}:${toString user.gid} data
    '';

    config = {
      User = user.name;
      Entrypoint = [ "${larql}/bin/larql-server" ];
      Cmd = [ "--help" ];
      ExposedPorts = {
        "8080/tcp" = {};     # HTTP (default)
        "50051/tcp" = {};    # gRPC (optional, enabled via --grpc-port)
      };
      WorkingDir = "/home/${user.name}";
      Volumes = {
        "/data" = {};
      };
      Env = [
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      ];
      Labels = {
        "org.opencontainers.image.source" = "https://github.com/chrishayuk/chuk-larql-rs";
        "org.opencontainers.image.description" = "LARQL server — query engine for transformer model weights";
        "org.opencontainers.image.licenses" = "Apache-2.0";
      };
    };
  };

  # ─── larql CLI Container ─────────────────────────────────────────────────
  cli = pkgs.dockerTools.buildLayeredImage {
    name = "larql";
    tag = "latest";

    contents = baseContents ++ [ larql ];

    extraCommands = ''
      mkdir -p home/${user.name}
      mkdir -p data
      mkdir -p tmp
      chmod 1777 tmp
    '';

    fakeRootCommands = ''
      chown -R ${toString user.uid}:${toString user.gid} home/${user.name}
      chown -R ${toString user.uid}:${toString user.gid} data
    '';

    config = {
      User = user.name;
      Entrypoint = [ "${larql}/bin/larql" ];
      Cmd = [ "--help" ];
      WorkingDir = "/home/${user.name}";
      Volumes = {
        "/data" = {};
      };
      Env = [
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      ];
      Labels = {
        "org.opencontainers.image.source" = "https://github.com/chrishayuk/chuk-larql-rs";
        "org.opencontainers.image.description" = "LARQL CLI — query engine for transformer model weights";
        "org.opencontainers.image.licenses" = "Apache-2.0";
      };
    };
  };

in
{
  inherit server cli;
}
