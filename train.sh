python train_CVAE.py --dataset "PoseUnit" \
                        --data_dir "./data" \
                        --epoch 50 \
                        --batch_size 14 \
                        --data_size -1 \
                        --z_dim 256 \
                        --y_dim 16 \
                        --pix_dim 256 \
                        --gpus 1 \
                        --worker 24 \
                        --lrG 1e-4 \
                        # --resume
