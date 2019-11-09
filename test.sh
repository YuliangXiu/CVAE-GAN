python starter.py --dataset "VAE-stretch" \
                        --data_dir "./data" \
                        --pkl "./checkpoint/VAE_data_VAE-stretch_pix_256_batch_24_embed_52/VAE-stretch" \
                        --batch_size 20 \
                        --data_size -1 \
                        --z_dim 52 \
                        --pix_dim 256 \
                        --gpus 0,1 \
                        --worker 48 \
                        --testmode
