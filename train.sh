python starter.py --dataset "VAE_NPY_stretch" \
                        --data_dir "./data" \
                        --epoch 5000 \
                        --batch_size 10 \
                        --data_size -1 \
                        --z_dim 52 \
                        --pix_dim 256 \
                        --gpus 0 \
                        --worker 24 \
                        --lrG 5e-5 \
                        # --resume
