"""Create the small, content-bound native empty-prompt cache."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

VAE_SHA = "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"
TEXT_SHA = "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"
MODELS = {
    "iris": (
        "f33104411b63e6d983e794386ece1feeddafd7bf",
        "dac6b77256db8d8fe35cb2e4b88d13bfd16b269eebe2b1310e9292e191ca4470",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def put(header: bytearray, offset: int, size: int, text: str) -> None:
    encoded = text.encode("ascii") + b"\0"
    if len(encoded) > size:
        raise ValueError("metadata field is too long")
    header[offset : offset + len(encoded)] = encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument(
        "--reuse-prompt-cache",
        type=Path,
        help=(
            "reuse an existing IRISP001 payload from the pinned Iris snapshot"
        ),
    )
    args = parser.parse_args()
    root = Path(args.snapshot)
    revision, unet_sha = MODELS[args.variant]
    files = {
        "UNet": (
            root / "unet" / "diffusion_pytorch_model.safetensors",
            unet_sha,
        ),
        "VAE": (
            root / "vae" / "diffusion_pytorch_model.safetensors",
            VAE_SHA,
        ),
        "text encoder": (
            root / "text_encoder" / "model.safetensors",
            TEXT_SHA,
        ),
    }
    for label, (path, expected) in files.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )

    if args.reuse_prompt_cache:
        source = args.reuse_prompt_cache.read_bytes()
        if (
            len(source) != 512 + 77 * 1024 * 4
            or source[:8] != b"IRISP001"
            or struct.unpack_from("<IIII", source, 8)
            != (1, 512, 77, 1024)
        ):
            raise RuntimeError("invalid source Iris prompt cache")
        prompt_payload = source[512:]
    else:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            root, subfolder="tokenizer", local_files_only=True
        )
        encoder = CLIPTextModel.from_pretrained(
            root, subfolder="text_encoder", local_files_only=True
        ).eval()
        tokens = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            prompt = encoder(
                tokens, return_dict=False
            )[0].contiguous().float()
        prompt_payload = prompt.numpy().tobytes()

    header = bytearray(512)
    header[:8] = b"IRISP001"
    struct.pack_into("<IIII", header, 8, 1, 512, 77, 1024)
    put(header, 24, 41, revision)
    put(header, 65, 65, unet_sha)
    put(header, 130, 65, VAE_SHA)
    put(header, 195, 65, TEXT_SHA)
    Path(args.output).write_bytes(header + prompt_payload)


if __name__ == "__main__":
    main()
