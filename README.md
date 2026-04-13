# innate-hackathon-egocentric-data-work

Cross-embodiment training pipeline for EgoVerse + MARS robot data.

## Scripts

| Script | Purpose |
|---|---|
| `build_dataset_gpu.py` | Download Scale episodes from R2 + convert to shared 6D/10D format with mp4s |
| `convert_to_shared_dataset.py` | Convert robot HDF5 episodes (joint angles) to shared representation via FK |
| `fast_webdataset.py` | Zero-copy parallel WebDataset converter |
| `shared_representation.py` | FK utilities, camera offset, URDF loading |
| `visualize_combined.py` | Interactive 3D MARS arm + human hand viewer |
| `data_viz_scale.py` | Colab notebook for Scale episode selection + export |

