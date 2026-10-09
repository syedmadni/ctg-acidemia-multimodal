# Table 5. Calibration of logistic regression at 60 min

| cohort | model | lr_weighting | n | positives | mean_predicted | prevalence | brier | brier_reference | cal_intercept | cal_intercept_boot_low | cal_intercept_boot_high | cal_slope | cal_slope_boot_low | cal_slope_boot_high | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckfilter | combined | class_weighted | 474 | 33 | 0.323 | 0.07 | 0.173 | 0.065 | -2.558 | -2.75 | -2.354 | 0.356 | 0.115 | 0.647 | 0.701 |
| ckfilter | combined | unweighted | 474 | 33 | 0.075 | 0.07 | 0.064 | 0.065 | -0.09 | -0.235 | 0.061 | 0.476 | 0.187 | 0.843 | 0.707 |
| ckfilter | ctg_only | class_weighted | 474 | 33 | 0.369 | 0.07 | 0.187 | 0.065 | -2.623 | -2.774 | -2.465 | 0.519 | 0.146 | 0.885 | 0.734 |
| ckfilter | ctg_only | unweighted | 474 | 33 | 0.071 | 0.07 | 0.063 | 0.065 | -0.031 | -0.148 | 0.087 | 0.666 | 0.284 | 1.082 | 0.722 |
| unfiltered | combined | class_weighted | 487 | 40 | 0.323 | 0.082 | 0.165 | 0.075 | -2.326 | -2.528 | -2.142 | 0.487 | 0.254 | 0.763 | 0.752 |
| unfiltered | combined | unweighted | 487 | 40 | 0.084 | 0.082 | 0.071 | 0.075 | -0.034 | -0.175 | 0.103 | 0.663 | 0.386 | 1.027 | 0.758 |
| unfiltered | ctg_only | class_weighted | 487 | 40 | 0.372 | 0.082 | 0.184 | 0.075 | -2.386 | -2.526 | -2.242 | 0.596 | 0.212 | 0.972 | 0.743 |
| unfiltered | ctg_only | unweighted | 487 | 40 | 0.085 | 0.082 | 0.07 | 0.075 | -0.041 | -0.157 | 0.079 | 0.774 | 0.442 | 1.154 | 0.745 |

Out-of-fold (repeat-averaged) predictions at 60 min. Intercept = calibration-in-the-large (GLM with offset logit p), slope = GLM of y on logit p; 2000 stratified bootstrap percentile CIs, treating OOF as fixed. Class-weighted = model as published (class_weight='balanced', which shifts predicted risk upward by design); unweighted = same pipeline without class weights. Ideal: intercept 0, slope 1.