```bash
uv sync --group examples
```

### 1. Fisher-Weighted Averaging on the GLUE dataset

```bash
USAGE: /examples/fisher_weightage.py [flags]

flags:
  --batch_size: The batch size to use
    (default: '2')
    (an integer)
  --model_ids: A comma-separated list of model ids to use
    (default: 'textattack/roberta-base-RTE,textattack/roberta-base-MNLI')
    (a comma separated list)
  --num_coefficients: The number of coefficients to use
    (default: '25')
    (an integer)
  --num_examples: The number of examples to use
    (default: '2048')
    (an integer)
  --task_name: The name of the glue task to use
    (default: 'rte')

🐕 ❯ python -m examples.fisher_weightage
computing fisher matrix: 100%|██████████████████████████████████████████████████████████████████| 139/139 [03:15<00:00,  1.41s/it]
computing fisher matrix: 100%|██████████████████████████████████████████████████████████████████| 139/139 [04:57<00:00,  2.14s/it]
trying different coefficients: 100%|██████████████████████████████████████████████████████████████| 25/25 [12:08<00:00, 29.14s/it]
Best result: {'accuracy': 0.7978339350180506}
Best coefficients: (0.16666666666666666, 0.8333333333333334)
```

#### References

* https://arxiv.org/abs/2111.09832 
* https://github.com/mmatena/model_merging