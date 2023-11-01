"""Extract foldK performance.

The model for fold K trained in 2021 was lost (probably delettion accident).
It was retrained while finishing up the Geophysics paper in November 2022.
Here we extract the performance numbers to update the paper.
"""

from hardpicks.metrics.fbp.evaluator import FBPEvaluator


pickle_path = (
    "/Users/bruno/monitoring/FBP/supplementary_experiments/"
    "supplement-predict/foldK/output_009bc585-7ab2-4ae9-b65d-fc52983552ad.best-epoch=017-step=025973_valid.pkl"
)

if __name__ == "__main__":
    evaluator = FBPEvaluator.load(pickle_path)
    results = evaluator.summarize()

    for key, value in results.items():
        if 'Error' in key:
            s = f'{value: 3.1f} pixels'
        else:
            s = f'{100 * value: 3.1f} %'
        print(f" {key} : {s}")
