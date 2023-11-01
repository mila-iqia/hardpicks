from hardpicks.metrics.fbp.evaluator import FBPEvaluator


def get_preconfigured_regression_evaluator():
    """Get an evaluator with the standard metrics for regression."""
    hr1 = dict(metric_type="HitRate", metric_params={"buffer_size_px": 1})
    hr3 = dict(metric_type="HitRate", metric_params={"buffer_size_px": 3})
    hr5 = dict(metric_type="HitRate", metric_params={"buffer_size_px": 5})
    hr7 = dict(metric_type="HitRate", metric_params={"buffer_size_px": 7})
    hr9 = dict(metric_type="HitRate", metric_params={"buffer_size_px": 9})
    mae = dict(metric_type="MeanAbsoluteError")
    rmse = dict(metric_type="RootMeanSquaredError")
    mbe = dict(metric_type="MeanBiasError")

    eval_metrics = [hr1, hr3, hr5, hr7, hr9, mae, rmse, mbe]
    config = dict(
        segm_class_count=None,  # set to NONE so that the predictions will be treated as picks, not logits
        segm_first_break_prob_threshold=0.0,
        eval_metrics=eval_metrics,
    )
    evaluator = FBPEvaluator(config)
    return evaluator
