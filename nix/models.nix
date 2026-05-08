#
# nix/models.nix - Model catalog for LARQL demos
#
# Each model is fetched from HuggingFace via git-lfs as a fixed-output
# derivation, fully reproducible and cached in /nix/store.
#
# To add a model:
#   1. Run: nix run nixpkgs#nix-prefetch-git -- --fetch-lfs https://huggingface.co/<org>/<model>
#   2. Copy the rev and hash into a new entry below
#   3. Gated models (Llama, Gemma 3) require HF authentication — Gemma 4 is ungated
#
# List available models:  nix run .#demo-list
# Run a specific model:   nix run .#demo -- --model qwen-0.5b "your prompt"
#
{ pkgs }:
let
  fetchModel = { name, url, rev, hash, size ? "unknown" }:
    {
      inherit name size;
      src = pkgs.fetchgit {
        inherit url rev hash;
        fetchLFS = true;
      };
    };
in
{
  # ─── Tiny Models (< 1GB safetensors) ─────────────────────────────────
  "smollm2-360m" = fetchModel {
    name = "SmolLM2 360M Instruct";
    url = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct";
    rev = "a10cc1512eabd3dde888204e902eca88bddb4951";
    hash = "sha256-POlS7POP/lLqTdnL3CoSLOtPqACPC2D1BVgyq3kfWAo=";
    size = "~725MB";
  };

  qwen-05b = fetchModel {
    name = "Qwen2.5 0.5B Instruct";
    url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct";
    rev = "7ae557604adf67be50417f59c2c2f167def9a775";
    hash = "sha256-Sg7nluvDsQhs0uBgcFtAwA5QErNC8tJFUkg6hiU353E=";
    size = "~970MB";
  };

  # ─── Small Models (1-3GB safetensors) ────────────────────────────────
  gemma4-2b = fetchModel {
    name = "Gemma 4 E2B Instruct";
    url = "https://huggingface.co/google/gemma-4-E2B-it";
    rev = "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf";
    hash = "sha256-CAaf8jR7FbGoHkcQVN/aoJi7FBpv7bEY7gDZFnNCIJc=";
    size = "~2.0GB";
  };

  tinyllama-1b = fetchModel {
    name = "TinyLlama 1.1B Chat v1.0";
    url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    rev = "fe8a4ea1ffedaf415f4da2f062534de366a451e6";
    hash = "sha256-vp/aUHKX+NJZZMIk2CgSh2czeGD0HeQGS30p/If2pA0=";
    size = "~2.2GB";
  };

  qwen-15b = fetchModel {
    name = "Qwen2.5 1.5B Instruct";
    url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct";
    rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306";
    hash = "sha256-YnR8BMkrPBXHs8RFbGVKiOG/Y2etyqOBlYm5t+kcNk8=";
    size = "~3.1GB";
  };

  stablelm-2b = fetchModel {
    name = "StableLM 2 Zephyr 1.6B";
    url = "https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b";
    rev = "2f275b1127d59fc31e4f7c7426d528768ada9ea4";
    hash = "sha256-qSsq2ZK5JqaA5VEFgbUgFgcb98bS4/7CWzB/COMjNro=";
    size = "~3.2GB";
  };

  # ─── Medium Models (3-8GB safetensors) ───────────────────────────────
  gemma4-4b = fetchModel {
    name = "Gemma 4 E4B Instruct";
    url = "https://huggingface.co/google/gemma-4-E4B-it";
    rev = "83df0a889143b1dbfc61b591bbc639540fd9ce4c";
    hash = "sha256-fEorwLDWBpJaN1QSBVHI6gtdmM48tKegSzMN7eWyjm4=";
    size = "~4.0GB";
  };

  phi-35-mini = fetchModel {
    name = "Phi-3.5 Mini Instruct (3.8B)";
    url = "https://huggingface.co/microsoft/Phi-3.5-mini-instruct";
    rev = "2fe192450127e6a83f7441aef6e3ca586c338b77";
    hash = "sha256-HnreQJ2iSEIFef5UxKrgYfbI3I0xlox38C3J6LJPgSE=";
    size = "~7.6GB";
  };

  # ─── Gated Models (require HF authentication) ───────────────────────
  # These are listed for reference but won't work without HF_TOKEN.
  # To add: accept license on HF website, set HF_TOKEN, then prefetch.
  #
  # gemma-3-1b = { ... };       # google/gemma-3-1b-it
  # llama-3.2-1b = { ... };     # meta-llama/Llama-3.2-1B-Instruct
}
