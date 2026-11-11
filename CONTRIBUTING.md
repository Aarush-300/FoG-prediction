# Contributing to Parkinson's FoG ESP32 Project

Thank you for your interest in improving this open-source wearable Parkinson's Disease intervention system.

## Ground Rules & Guidelines
1. **Strictly Subject-Independent Validation:** Never introduce test set metrics into hyperparameter tuning, model architecture selection, or threshold calibration.
2. **Physical Sensor Units:** All sensor data preprocessing and augmentations must operate on raw physical units ($g$ and $\text{deg/s}$) before normalization.
3. **Reproducibility:** Fix random seeds across all experiments (`set_seed(42)`).
4. **Unit Testing:** Ensure all tests pass prior to submitting a PR:
   ```bash
   pytest tests/ -v
   ```
5. **Firmware Alignment:** If changing model architectures or scaler parameters, ensure `firmware/ESP32_FoG/model_data.h`, `model_config.h`, and `golden_vectors.h` are re-exported and tested with `python src/verify_esp32_deployment.py`.
