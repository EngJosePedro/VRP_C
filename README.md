

Train comand:
python train.py --graph_size 20 --batch_size 256 --epoch_size 128000 --pomo 2

Eval:
python eval.py --model outputs/vrp_100/epoch-99.pt --dataset "instance\datasets_pkl\X_93*.pkl" --eval_batch_size 8 --pomo 128


pretreined models are in Releases.
