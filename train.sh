python starter.py --dataset "PoseRandom-stretch" \
                        --data_dir "./data" \
                        --epoch 6000 \
                        --batch_size 48 \
                        --data_size -1 \
                        --z_dim 52 \
                        --y_dim 51 \
                        --pix_dim 256 \
                        --gpus 0,1,2,3 \
                        --worker 64 \
                        --lrG 3e-5 \
                        # --resume
