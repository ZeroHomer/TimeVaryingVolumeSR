nohup python -u train.py --train_data_dir ionization_ab_H/train --val_data_dir ionization_ab_H/test --dim "(96, 96, 144)" --dataset_name ionization_ab_H --train_batch_size 3 --n_epochs 600 --lambda_content 0 --ssim --cuda --attn --resume /home/dell/YeKaiwei/SuperResolution/checkpoint_ConvGRUVSR_ionization_ab_H_no_content/ConvGRUVSR_epoch_52.pth >> ionization_no_content.log &