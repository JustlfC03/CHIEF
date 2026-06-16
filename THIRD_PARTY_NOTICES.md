# Third-party notices

CHIEF v2.0.0 is a cleaned research implementation. No third-party model weights, clinical data, private manifests or external source repositories are bundled in this archive.

## Runtime libraries and model interfaces

The project depends on open-source packages listed in `requirements.txt`. Each package remains governed by its own copyright notice and licence.

The full configurations reference `hfl/chinese-bert-wwm-ext` and `uer/gpt2-chinese-cluecorpussmall`. Their weights are downloaded separately and remain subject to the terms published by their distributors.

## Vector quantization

The original active CTViT path imported `VectorQuantize` from `vector-quantize-pytorch`. CHIEF v2.0.0 pins `vector-quantize-pytorch==1.1.2` and uses that upstream implementation directly, including cosine-similarity codebooks. The package is not vendored into this repository.

## Volumetric transformer lineage

The original research directory contained a volumetric transformer under `transformer_maskgit` and identified GenerateCT as an architectural reference. The cleaned CTViT module retains only the components required by CHIEF. Users redistributing modified versions should review and preserve any upstream notices that apply to code they additionally incorporate.

The presence of a project or package name in this notice does not imply endorsement by its authors.
