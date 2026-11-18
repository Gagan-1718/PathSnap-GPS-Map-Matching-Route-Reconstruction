# Dataset

The raw data is not committed because it is about 90 MB. Put the provided dataset here:

```
data/
├── train/          # 80 trips: used for training and cross-validation
│   └── 00000001/   # 00000001.track  .nodes  .arcs  .route
└── student_test/   # 10 trips: used only for comparing models
```

The evaluator-testing trips are never placed in this folder during development.
