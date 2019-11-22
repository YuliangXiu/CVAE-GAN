python starter.py --dataset "L1_CLS_patchD" \
                        --data_dir "./data" \
                        --epoch 5000 \
                        --batch_size 20 \
                        --data_size -1 \
                        --z_dim 51 \
                        --pix_dim 256 \
                        --gpus 0,1 \
                        --worker 48 \
                        --lrG 1e-4 \
                        # --resume
