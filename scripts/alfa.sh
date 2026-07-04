

for f in ./dataset/alfa/test/*.csv; do
  ALFA_TEST_FILE="$(basename "$f")" \
  python run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/ALFA \
    --data AlfaRecon \
    --model VAE_LSTM \
    --seq_len 48 \
    --enc_in 18 \
    --c_out 18 \
    --features M
done