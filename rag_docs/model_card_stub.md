# Model Card Notes

The fraud model uses transaction amount, transaction type, balance deltas, and time-safe customer
history features. The target is is_fraud. The is_flagged_fraud field is excluded from model features
because it represents an existing business rule.
