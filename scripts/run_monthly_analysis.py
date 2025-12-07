{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02605038-df1a-4462-ab42-f6aa157a1a43",
   "metadata": {},
   "outputs": [],
   "source": [
    "# scripts/run_monthly_analysis.py\n",
    "from src.io_monthly import load_monthly_gold_csv\n",
    "from src.diagnostics import acf_pacf_plots, stl_decompose, prepare_stationary\n",
    "from src.regimes_vol import plot_regime_vol, zscore_regimes\n",
    "from src.baselines import time_split, fit_sarima_forecast, fit_holt_winters_forecast, evaluate_forecast, plot_forecasts\n",
    "\n",
    "def main():\n",
    "    df_m = load_monthly_gold_csv(\"XAU_1Month_data.csv\")\n",
    "    close_m = df_m[\"close\"].dropna()\n",
    "\n",
    "    # 1) Trend/seasonality diagnostics\n",
    "    stl_decompose(close_m, period=12, title=\"Monthly Gold Close STL (12-month seasonality)\")\n",
    "    acf_pacf_plots(prepare_stationary(close_m, use_log=True, diff_order=1),\n",
    "                   max_lag=48, title_prefix=\"Monthly Gold (log diff)\")\n",
    "\n",
    "    # 2) Cycles/regimes/volatility\n",
    "    plot_regime_vol(close_m, windows=(12, 24), title=\"Regime and volatility overview (monthly)\")\n",
    "    reg_df = zscore_regimes(close_m, window=24, threshold=1.0)\n",
    "    print(\"Recent regime flags:\\n\", reg_df.tail())\n",
    "\n",
    "    # 3) Baselines and benchmarking\n",
    "    train, test = time_split(close_m, test_size=24)\n",
    "    sarima_res, sarima_pred = fit_sarima_forecast(train, test, order=(1,1,1), seasonal_order=(0,1,1,12))\n",
    "    hw_res, hw_pred = fit_holt_winters_forecast(train, test, trend=\"add\", seasonal=\"add\", seasonal_periods=12)\n",
    "\n",
    "    sarima_metrics = evaluate_forecast(test, sarima_pred)\n",
    "    hw_metrics = evaluate_forecast(test, hw_pred)\n",
    "    print(\"SARIMA:\", sarima_metrics)\n",
    "    print(\"Holt-Winters:\", hw_metrics)\n",
    "\n",
    "    plot_forecasts(train, test, preds={\"SARIMA\": sarima_pred, \"Holt-Winters\": hw_pred},\n",
    "                   title=\"Monthly gold forecast benchmarks\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
