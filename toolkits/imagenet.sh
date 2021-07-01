python -m torch.distributed.launch --nproc_per_node=4 tools/baseline_imagenet.py -a resnet50 --b 224 --workers 4 --opt-level O1 /gpub/imagenet_raw

