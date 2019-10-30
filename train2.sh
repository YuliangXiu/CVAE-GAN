python train_CVAE.py --dataset "PoseUnit-stretch" \
                        --data_dir "./data" \
                        --epoch 50 \
                        --batch_size 14 \
                        --data_size -1 \
                        --z_dim 64 \
                        --y_dim 12 \
                        --pix_dim 256 \
                        --gpus 0 \
                        --worker 24 \
                        --lrG 1e-4 \
                        # --resume
