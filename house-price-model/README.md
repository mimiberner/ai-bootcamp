# House price model benchmark

Compares seven regression models for predicting the median house value of a California census district (California Housing dataset, 1990 US Census, 20,640 districts).

## Results

Test set: 20% of rows (4,128 districts), never seen during training.

| Model | Test RMSE | R² | Within ±10% | Training time |
|---|---|---|---|---|
| LightGBM | $44,248 | 0.851 | 51% | 1.2 s |
| Stacked ensemble (RF + XGBoost + LightGBM) | $44,405 | 0.850 | 51% | 21 s |
| XGBoost | $46,096 | 0.838 | 48% | 0.7 s |
| Random forest | $49,794 | 0.811 | 46% | 2.2 s |
| Decision tree | $58,264 | 0.741 | 38% | 0.1 s |
| Neural network (MLP) | $59,997 | 0.725 | 34% | 2.3 s |
| Linear regression | $67,500 | 0.652 | 29% | <0.1 s |

Open `report.html` in a browser for the full report: all KPIs (MAE, median error, MAPE, cross-validation, overfit gap), predicted-vs-actual plots and feature importance.

## Run it

```bash
pip install -r requirements.txt
python train_models.py   # trains all models (~30 s), writes results.json
python build_report.py   # writes report.html from report_template.html + results.json
```

## Approach

- Three engineered features: rooms per household, bedrooms per room, people per household.
- All models predict log(price), and predictions are clipped to the price range in the data.
- Linear regression and the neural network get log-compressed, standardised inputs.
- Models are kept small so the whole benchmark trains in under a minute.

## Caveats

- 4.8% of districts sit at the census cap of $500,001, so no model can predict above it.
- The data is from 1990, so absolute prices are far below today's.
