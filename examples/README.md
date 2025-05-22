```bash
uv sync --group examples
```

### 1. Fisher-Weighted Averaging on the GLUE dataset

[![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/sauravmaheshkar/fisher-merging/reports/glue-validation-set-accuracy-using-different-ensembling-method--VmlldzoxMjg2MTA0Ng)

```bash
USAGE: /examples/fisher_weightage.py [flags]

flags:
  --batch_size: The batch size to use
    (default: '2')
    (an integer)
  --model_ids: A comma-separated list of model ids to use
    (default: 'textattack/bert-base-uncased-RTE,yoshitomo-matsubara/bert-base-uncased-rte')
    (a comma separated list)
  --num_coefficients: The number of coefficients to use
    (default: '25')
    (an integer)
  --num_examples: The number of examples to use
    (default: '2048')
    (an integer)
  --task_name: The name of the glue task to use
    (default: 'rte')
  --[no]use_wandb: Whether to use wandb
    (default: 'false')
  --wandb_entity: The wandb entity to use
    (default: os.getenv("WANDB_ENTITY"))
  --wandb_project: The wandb project to use
    (default: os.getenv("WANDB_PROJECT"))

🐕 ❯ python -m examples.fisher_weightage
Output ensembling result: {'accuracy': 0.7075812274368231}

computing fisher matrix: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 139/139 [02:49<00:00,  1.22s/it]
computing fisher matrix: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 139/139 [02:40<00:00,  1.16s/it]
trying different coefficients: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 25/25 [11:08<00:00, 26.75s/it]

Best fisher result: {'accuracy': 0.7472924187725631}
Best fisher coefficients: (0.5833333333333334, 0.41666666666666663)

trying different coefficients: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 25/25 [10:50<00:00, 26.04s/it]

Best isotropic result: {'accuracy': 0.7256317689530686}
Best isotropic coefficients: (0.875, 0.125)
```

#### References

* https://arxiv.org/abs/2111.09832 
* https://github.com/mmatena/model_merging
