
ALFA_TEST_FILE="carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv" \
python run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --model_id alfa_recon_lstm_ae_debug \
  --model LSTM_AE \
  --data AlfaRecon \
  --root_path data/alfa \
  --seq_len 48 \
  --pred_len 0 \
  --label_len 0 \
  --enc_in 18 \
  --dec_in 18 \
  --c_out 18 \
  --features M \
  --d_model 128 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1 \
  --batch_size 64 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate 1e-4 \
  --num_workers 0 \
  --anomaly_ratio 1 \
  --checkpoints /tmp/xlstmflow_checkpoints


# python run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --model_id alfa_recon_debug \
#   --model TimesNet \
#   --data AlfaRecon \
#   --root_path data/alfa/ \
#   --seq_len 48 \
#   --pred_len 0 \
#   --label_len 0 \
#   --enc_in 18 \
#   --dec_in 18 \
#   --c_out 18 \
#   --features M \
#   --batch_size 64 \
#   --train_epochs 30 \
#   --patience 2 \
#   --learning_rate 1e-4 \
#   --num_workers 0 \
#   --anomaly_ratio 1

# for f in ./dataset/alfa/test/*.csv; do
#   ALFA_TEST_FILE="$(basename "$f")" \
#   python run.py \
#     --task_name anomaly_detection \
#     --is_training 1 \
#     --root_path ./dataset/ALFA \
#     --data AlfaRecon \
#     --model VAE_LSTM \
#     --seq_len 48 \
#     --enc_in 18 \
#     --c_out 18 \
#     --features M
# done