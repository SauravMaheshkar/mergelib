import gc
import os

import evaluate
import torch
from absl import app, flags
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mergelib import merge

from .glue_utils import load_glue_dataset


load_dotenv()


FLAGS = flags.FLAGS

flags.DEFINE_list(
    name="model_ids",
    default=[
        "textattack/bert-base-uncased-RTE",
        "yoshitomo-matsubara/bert-base-uncased-rte",
    ],
    help="A comma-separated list of model ids to use",
)
flags.DEFINE_string(
    name="task_name", default="rte", help="The name of the glue task to use"
)
flags.DEFINE_integer(
    name="num_examples", default=2048, help="The number of examples to use"
)
flags.DEFINE_integer(
    name="num_coefficients",
    default=25,
    help="The number of coefficients to use",
)
flags.DEFINE_integer(name="batch_size", default=2, help="The batch size to use")
flags.DEFINE_bool(
    name="use_wandb",
    default=False,
    help="Whether to use wandb",
)
flags.DEFINE_string(
    name="wandb_entity",
    default=os.getenv("WANDB_ENTITY"),
    help="The wandb entity to use",
)
flags.DEFINE_string(
    name="wandb_project",
    default=os.getenv("WANDB_PROJECT"),
    help="The wandb project to use",
)


def main(_):
    if FLAGS.use_wandb:
        import wandb

        wandb.init(
            entity=FLAGS.wandb_entity,
            project=FLAGS.wandb_project,
            config=FLAGS.flag_values_dict(),
        )

    models = [
        AutoModelForSequenceClassification.from_pretrained(model_id)
        for model_id in FLAGS.model_ids
    ]
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_ids[0])

    # ======== load dataset ========
    ds = load_glue_dataset(FLAGS.task_name, "validation", tokenizer, max_length=128)
    ds = ds.with_format("torch")
    ds = ds.take(min(FLAGS.num_examples, len(ds))).batch(FLAGS.batch_size)

    # ======== load metric ========
    metric = evaluate.load("glue", FLAGS.task_name)

    # ======== output ensembling ========
    for batch in ds:
        logits_list = []

        for model in models:
            outputs = model(**batch)
            logits_list.append(outputs.logits)

        stacked_logits = torch.stack(logits_list, dim=0)
        avg_logits = torch.mean(stacked_logits, dim=0)
        preds = torch.argmax(avg_logits, dim=-1)
        y = batch["labels"]
        metric.add_batch(predictions=preds, references=y)

    ensemble_result = metric.compute()
    print(f"Output ensembling result: {ensemble_result}\n")

    _ = gc.collect()

    # ======== fisher merge ========
    metric = evaluate.load("glue", FLAGS.task_name)
    _fisher_merged_model, best_fisher_result, best_fisher_coeff = merge(
        models,
        ds,
        metric,
        coefficient_type="grid",
        method="fisher",
        num_coefficients=FLAGS.num_coefficients,
    )

    print(f"\nBest fisher result: {best_fisher_result}")
    print(f"Best fisher coefficients: {best_fisher_coeff}\n")

    _ = gc.collect()

    # ======== isotropic merge ========
    metric = evaluate.load("glue", FLAGS.task_name)
    _isotropic_merged_model, best_isotropic_result, best_isotropic_coeff = merge(
        models,
        ds,
        metric,
        coefficient_type="grid",
        method="isotropic",
        normalize=False,
        num_coefficients=FLAGS.num_coefficients,
    )

    print(f"\nBest isotropic result: {best_isotropic_result}")
    print(f"Best isotropic coefficients: {best_isotropic_coeff}")

    _ = gc.collect()

    if FLAGS.use_wandb:
        data = [
            ["fisher-merge", best_fisher_result["accuracy"]],
            ["isotropic-merge", best_isotropic_result["accuracy"]],
            ["output-ensembling", ensemble_result["accuracy"]],
        ]
        table = wandb.Table(data=data, columns=["model", "accuracy"])
        wandb.log(
            {
                "accuracy-table": wandb.plot.bar(
                    table, "model", "accuracy", title="Accuracy Comparison"
                )
            }
        )
        wandb.summary["best-fisher-coefficients"] = best_fisher_coeff
        wandb.summary["best-isotropic-coefficients"] = best_isotropic_coeff
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
