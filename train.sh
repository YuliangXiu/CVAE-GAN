python train_CVAE-GAN.py --dataset "GT" \
                        --data_dir "./data" \
                        --epoch 10 \
                        --batch_size 14 \
                        --z_dim 64 \
                        --y_dim 10 \
                        --pix_dim 256 \
                        --gpus 0 \
                        --worker 12 \
                        --lrG 1e-4