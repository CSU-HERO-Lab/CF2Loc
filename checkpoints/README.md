# Checkpoints

Model weights are intentionally excluded from Git. Place the pretrained
backbone and release checkpoints at the following paths:

```text
checkpoints/depth_anything_v2_vits.pth
checkpoints/release_s3d/s3d_no_sem_stage1_best.ckpt
checkpoints/release_s3d/s3d_no_sem_dense_refiner_best.ckpt
checkpoints/release_s3d/s3d_semantic_onehot_stage1_best.ckpt
checkpoints/release_s3d/s3d_semantic_onehot_dense_refiner_best.ckpt
checkpoints/release_zind/zind_no_sem_stage1_best.ckpt
checkpoints/release_zind/zind_no_sem_dense_refiner_best.ckpt
checkpoints/release_zind/zind_semantic_onehot_stage1_best.ckpt
checkpoints/release_zind/zind_semantic_onehot_dense_refiner_best.ckpt
```

The manifests in `release_s3d` and `release_zind` record the expected SHA-256
hashes and the validation criteria used to select each checkpoint.
