import numpy as np

import hardpicks.data.fbp.gather_wrappers as wrappers


class FakeWhateverDataset:

    def __init__(self, size: int, root: str):
        self.data = [dict(idx=idx, root=root) for idx in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "samples": np.random.rand(10, 10),
            **self.data[item],
        }

    def get_meta_gather(self, item):
        return self.data[item]


def test_concat_dataset():
    dataset1 = FakeWhateverDataset(100, root="dataset1")
    dataset2 = FakeWhateverDataset(50, root="dataset2")
    dataset3 = FakeWhateverDataset(1, root="dataset3")

    dataset_glob = wrappers.ShotLineGatherConcatDataset([dataset1, dataset2, dataset3])
    assert len(dataset_glob) == 151
    full_sample = dataset_glob[0]
    assert "samples" in full_sample and "idx" in full_sample and "root" in full_sample
    assert full_sample["idx"] == 0 and full_sample["root"] == "dataset1"

    counts = {"dataset1": 0, "dataset2": 0, "dataset3": 0}
    max_idx = {"dataset1": 100, "dataset2": 50, "dataset3": 1}
    for idx in range(len(dataset_glob)):
        meta = dataset_glob.get_meta_gather(idx)
        counts[meta["root"]] += 1
        assert max_idx[meta["root"]] > meta["idx"]
    assert counts["dataset1"] == 100
    assert counts["dataset2"] == 50
    assert counts["dataset3"] == 1

    weights = dataset_glob.get_sample_weights()
    assert len(weights) == len(dataset_glob)
    assert all([w == 151 / 100 for w in weights[:100]])
    assert all([w == 151 / 50 for w in weights[100:150]])
    assert weights[-1] == 151


def test_dataset_subset():
    full_dataset = FakeWhateverDataset(1000, root="whatevs")
    subset_len = 100
    idxs = np.random.permutation(len(full_dataset))[:subset_len]
    subset_dataset = wrappers.ShotLineGatherSubset(full_dataset, idxs)
    assert len(subset_dataset) == subset_len
    for subset_idx, real_idx in enumerate(idxs):
        subset_meta = subset_dataset.get_meta_gather(subset_idx)
        subset_full = subset_dataset[subset_idx]
        assert subset_meta["idx"] == subset_full["idx"] and subset_full["idx"] == real_idx
