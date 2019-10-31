python train_CVAE.py --dataset "PoseUnit-stretch" \
                        --data_dir "./data" \
                        --epoch 50 \
                        --batch_size 12 \
                        --data_size -1 \
                        --z_dim 256 \
                        --y_dim 16 \
                        --pix_dim 256 \
                        --gpus 0 \
                        --worker 24 \
                        --lrG 1e-5 \
                        # --resume
