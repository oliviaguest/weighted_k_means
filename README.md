# Weighted _K_-means

To Run your own weighted _k_-means use ```example.py``` and refer to ```wkmeans.py```'s source code which is fully commented.

## Details

We use a distance scaling factor per cluster based on Luce's choice axiom. Small imbalances in cardinality, leading to huge imbalances, which further lead to empty clusters, must be avoided. Thus to ensure that the algorithm is stable between runs, the scaling factor is time-averaged. More details will be added soon.
