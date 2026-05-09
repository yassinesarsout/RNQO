# ─────────────────────────────────────────────────────────────────
# ADD THIS TO EACH TRAINING NOTEBOOK — right after model.fit()
# The dashboard's forecast engine requires both the model AND the
# scaler to be saved next to each other in "my training code/"
# ─────────────────────────────────────────────────────────────────

import joblib

# ── For the DL PRB Load notebook ──────────────────────────────────
joblib.dump(scaler, "prb_scaler.pkl")
model.save("prb_model.keras")

# ── For the Active Users notebook ─────────────────────────────────
joblib.dump(scaler, "active_users_scaler.pkl")
model.save("active_users_model.keras")

# ── For the LTE DL User Throughput notebook ───────────────────────
joblib.dump(scaler, "throughput_scaler.pkl")
model.save("throughput_model.keras")

# Install joblib if needed:  pip install joblib
