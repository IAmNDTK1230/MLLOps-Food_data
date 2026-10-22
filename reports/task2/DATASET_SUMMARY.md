# Dataset sources and frozen split

VN-MSASar v3 has 76,785 source rows, filtered to 5,869 Vietnamese restaurant sentence/aspect rows and 1,530 unique reviews. The frozen split is train 4,140, validation 842, test 887; seed 42, split SHA256 `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`. The connected-component split groups review IDs and normalized duplicate text; both reported cross-split counts are zero. Fuzzy near-duplicate leakage is not ruled out. Labels are supplied by the dataset.

The first-party Foody export has 33,946 raw rows, 33,938 valid AI rows, 32,457 distinct valid review IDs, and 2,110 restaurants. Foody has no trusted gold sentiment/aspect labels and is used for inference/demo/RAG, not held-out research scoring. The export has no documented public URL. See `dataset_sources.csv` and upstream source manifests.
