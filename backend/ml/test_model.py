from bitalino_model import train_model_from_folders, score_file

# Train on baseline (CALM) files
model = train_model_from_folders("CALM/*.txt")

# Score one stress file
avg_score, anomaly_score, preds = score_file(model, "STRESS/stress_andi_ecg_eda.txt")

print("Average anomaly score:", avg_score)
print("Preds (1 normal, -1 anomaly):", preds[:20])