python starter.py --dataset "VAE-stretch" \
                        --data_dir "./data_unit" \
                        --pkl "./checkpoint/VAE_NPY_stretch_pix_256_batch_12_embed_52/VAE_NPY_stretch" \
                        --batch_size 20 \
                        --data_size -1 \
                        --z_dim 52 \
                        --pix_dim 256 \
                        --gpus 1 \
                        --worker 48 \
                        --testmode
