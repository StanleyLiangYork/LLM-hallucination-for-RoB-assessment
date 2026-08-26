# Data directory

Expected local layout:

```text
data/
├── CD014722-data-rows.csv
├── human_review_structured/
│   └── <Topic>/<Study>/<Study>.json
└── pdfs/
    ├── Acarturk 2022/
    │   └── Acarturk_2022.pdf
    └── ...
```

`CD014722-data-rows.csv` contains the Cochrane review data rows and human RoB 2 support/judgement fields used by `extract_human_rob.py`.

`human_review_structured/` contains the deterministic extraction generated from that CSV. Its manifest identifies all 246 unique study-topic records and the source CSV row numbers merged into each record.

The trial-report PDFs and the Cochrane review PDF are third-party materials. They are intentionally excluded by `.gitignore` and must not be redistributed unless their licenses permit it. Users should obtain source reports from the original publishers or the Cochrane Library and place them in the local folder structure above.

API keys must never be stored under `data/` or committed to version control.
