python starter.py --dataset "L1_CLS_patchD" \
                        --data_dir "./data" \
                        --pkl "./checkpoint/L1_CLS_patchD_pix_256_batch_22_embed_51/L1_CLS_patchD" \
                        --batch_size 5 \
                        --data_size -1 \
                        --z_dim 51 \
                        --pix_dim 256 \
                        --gpus 1 \
                        --worker 48 \
                        --testmode
