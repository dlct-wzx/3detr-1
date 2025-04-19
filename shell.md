# Trainting
```shell
python main.py --dataset_name to_scene --nqueries 256 --meta_data_dir to_scene
```

# Testing
```shell
python main.py --dataset_name to_scene --nqueries 256 --test_ckpt ./log_2scene_arg/checkpoint_best.pth --test_only  --meta_data_dir to_scene --batchsize_per_gpu 24 --visualize False
```