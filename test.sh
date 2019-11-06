python starter.py --dataset "PoseRandom-stretch" \
                        --data_dir "./data" \
                        --pkl "./checkpoint/VAE_data_PoseRandom-stretch_pix_64_batch_400_embed_52_label_51/PoseRandom-stretch" \
                        --batch_size 20 \
                        --data_size -1 \
                        --z_dim 52 \
                        --y_dim 51 \
                        --pix_dim 64 \
                        --gpus 0,1 \
                        --worker 48 \
                        --testmode
