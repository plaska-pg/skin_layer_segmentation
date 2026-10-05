# psuedolabels
these are all images with good labeleds. 

![
    ![alt text](S_EX_9_24_D2_L1C2.jpg)
    ![alt text](S_EX_9_24_D2_L1C3.jpg)
    ![alt text](S_EX_9_24_D2_L7C1.jpg)
    ![alt text](S_EX_9_24_D2_L7C2.jpg)
    ![alt text](S_EX_9_24_D2_L7C4.jpg)
    ![alt text](S_EX_9_24_D2_L8C1.jpg)
    ![alt text](S_EX_9_24_D2_L8C1-42Y5X84-W10s.jpg)
    ![alt text](S_EX_9_24_D2_L8C2.jpg)
    S_EX_9_24_D2_L8C3
    ![alt text](S_EX_9_24_D2_L10C3.jpg)
    ![alt text](S_EX_9_24_D2_L10C4.jpg)
    S_EX_9_24_D2_L11C1
](S_EX_9_24_D2_L1C1.jpg)

# view training data:
.\.venv\Scripts\python.exe show_train_samples.py --pseudo-only --count 30 --seed 1

# to train: 
.\.venv\Scripts\python.exe -u pretrained_model\train_semseg.py train `
  --data dataset_pseudolabel_finetune `
  --init-weights runs\semseg\dataset_2cls_finetune_v2\best.pt `
  --pretrained-classes "SC,Granular Layer" `
  --encoder resnet34 --epochs 60 --batch 4 --crop 512 --lr 1e-4 `
  --val-every 10 --patience 5 --workers 2 --device auto `
  --project runs\semseg --name dataset_2cls_pseudo_v1