python train.py --batch-size 4 --data data/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7_training.pt' --name struct_det --hyp data/hyp.scratch.p5.yaml --epochs 35
python test.py --data data/data.yaml --img 640 --batch 4 --conf 0.3 --iou 0.65 --device 0 --weights runs/train/struct_det/weights/best.pt --name level7
