for i in $(seq 1 3); do
    echo "========== The $i Trainning =========="
    python train_university.py \
        --epochs=120 \
        --classes_num=35 \
        --sample_num=5 \
        --lr=0.001 \
        --scheduler='cosine' \
        --warmup_epochs=5
    sleep 60
done

















