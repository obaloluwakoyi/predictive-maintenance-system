# Predictive Maintenance System for Offshore Compressor Units

An enterprise machine learning pipeline built to monitor continuous telemetry streams from complex heavy machinery and forecast catastrophic failures up to 12 hours before mechanical degradation occurs.

## 🚀 Business Impact & Value
Unplanned asset maintenance costs offshore energy operations massive capital daily. This system serves as an early warning trigger that shifts engineering infrastructure from *reactive firefighting* to *optimized proactive schedules*.

## 🛠️ Pipeline Architecture
1. **Sanitization Engine:** 3x Interquartile Range (IQR) bounding logic to nullify erratic electrical grid sensor spikes.
2. **Feature Extractor:** Expansion of raw time-series metrics into 24 rolling-window and delayed lag vectors capturing deep structural vibrations.
3. **Optimized Core:** Imbalance-aware XGBoost tuned using TimeSeriesSplit Cross-Validation maximizing F1-Score metrics rather than misleading basic accuracy.

## 📊 Key Insights
- The system achieves reliable failure classification by prioritizing historical rolling standard deviation matrices over simple instantaneous sensor readouts.
- **Vibration 12-Hour Rolling Standard Deviation** emerged as the top leading indicator for bearing failure signatures.
