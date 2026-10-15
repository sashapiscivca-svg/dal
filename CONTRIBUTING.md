# Contributing to dal

Thank you for helping. dal improves most from **real-world evidence**: footage where it loses people, measurements on small hardware, and openly licensed annotated tracks. Code is welcome too.

## Ways to help

- **Report where the tracker fails.** Open an issue with the profile, camera (RGB or thermal), altitude, frame rate, the initial box and a few frames around the failure. Sensitive footage can be reported privately (see [SECURITY.md](SECURITY.md)).
- **Benchmarks on hardware:** run `dal bench short` on a prepared set and the tracker on your board, and send the timings with the board, OS, Python and OpenCV versions.
- **Annotated tracks under an open licence** (CC0, CC BY or MIT; civilian; with the consent required in your jurisdiction). Hand-labelled long thermal tracks of people on land are what the field misses most. We cannot accept data whose licence forbids commercial use or derivatives.
- **Code:** bug fixes, speed-ups for weak CPUs, better loss detection, documentation and translations.

## Ground rules

1. **Civilian scope.** Contributions that serve the uses listed as out of scope in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) will not be merged. This includes vehicle, gimbal or camera control, "follow the target" logic, geolocation of tracked people, and any re-identification across videos or sessions.
2. **No third-party trained weights.** The learned model is trained from scratch so that its licence stays clean.
3. **Licence-clean data only.** Check the licence of the data itself on its primary source and record it in `docs/datasets.md`.
4. **Measure on held-out data.** A change that claims better tracking needs numbers from `dal bench` on clips that were not used to choose it (`--every 3 --offset 1/2`, or other locations), and on at least two datasets. Many ideas in this project won on tuning clips and lost on held-out ones.
5. **Keep the runtime light.** `dal.classic`, `dal.bench` and everything they import must work with numpy and OpenCV only. PyTorch is for the learned path.
6. **Defaults change only with evidence.** New mechanisms go in as options that are off by default.

## Development setup

```bash
git clone https://github.com/sashapiscivca-svg/dal && cd dal
pip install -e ".[dev,bench,onnx,train,figures]"
pytest -q                  # all tests run on CPU with synthetic data; no datasets needed
ruff check src tests scripts
```

## Pull requests

- Keep each PR focused, and add or update tests.
- Code and comments are in English. User documentation is maintained in English, and the README also in Ukrainian; if you change one README, say so in the PR and we will sync the other.
- Markdown: one line per paragraph or list item (no manual line wrapping).
- Describe what changed and how you verified it (commands and numbers).
- By contributing, you agree that your contribution is licensed under the MIT License of this project.
