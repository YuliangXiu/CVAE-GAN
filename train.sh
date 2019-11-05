python starter.py --dataset "PoseRandom-stretch" \
                        --data_dir "./data" \
                        --epoch 10000 \
                        --batch_size 550 \
                        --data_size -1 \
                        --z_dim 32 \
                        --y_dim 51 \
                        --pix_dim 64 \
                        --gpus 0,1 \
                        --worker 48 \
                        --lrG 1e-4 \
                        # --resume
