# Security and safety reporting

## What to report

- **Security vulnerabilities** in the code, for example path traversal in the command line or dataset scripts, or unsafe handling of downloaded files.
- **Safety issues**: systematic failure modes that could cost lives (a kind of footage, a camera or a terrain where people are reliably lost, or a confident box held on empty background), wrong defaults, or published numbers you cannot reproduce.
- **Misuse** of dal that conflicts with [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## How to report

Please use **GitHub private vulnerability reporting** on this repository ("Security" tab → "Report a vulnerability"). This reaches the maintainers privately. Do not open a public issue for vulnerabilities or misuse reports. Tracking-quality issues that are not sensitive can go to normal GitHub issues.

Please include the version (`dal --version`), the tracker and profile, steps or data to reproduce (a few frames and the initial box are enough), and what you expected.

We aim to acknowledge reports within 7 days and to publish a fix or an advisory within 90 days. If you report a safety issue, we will credit you in the changelog unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Verifying model files

The released dal-n files are listed with their SHA-256 in `models/SHA256SUMS`:

```bash
cd models && sha256sum -c SHA256SUMS
```

The released formats (ONNX and safetensors) cannot execute code when loaded. Training checkpoints (`.pt`) are loaded with `torch.load(..., weights_only=True)`; still, load only `.pt` files that you produced yourself.
