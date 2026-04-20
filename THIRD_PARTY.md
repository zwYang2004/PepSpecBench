# Third-party dependencies (minimal note)

PepSpecBench uses a unified benchmark interface but some baseline runners expect external projects.
This package intentionally does not bundle full third-party repositories.

## Upstream projects referenced by current runners
- Prosit dependencies (`dlomix`, `prosit`-related utilities)
- Prosit Transformer dependencies (`prosittransformer`, `tape` utilities)
- PredFull (`PredFull`)
- AlphaPeptDeep (`alphapeptdeep`)
- UniSpec (`UniSpec`)
- FastSpel (`fastspel`)

## Release policy
- Use official upstream repositories/packages and pin versions in your environment.
- Keep local wrapper/adapter code in this repository under `src/models/runners`.
- Do not copy proprietary or unclear-license assets into this release package.

## License reminder
Before public release, add exact upstream URLs + version/commit + license names for each dependency in this file.
