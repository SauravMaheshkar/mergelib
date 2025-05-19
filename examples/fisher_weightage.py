import evaluate
from absl import app, flags
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mergelib import merge

from .glue_utils import load_glue_dataset


FLAGS = flags.FLAGS

flags.DEFINE_list(
    name="model_ids",
    default=["textattack/roberta-base-RTE", "textattack/roberta-base-MNLI"],
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


def main(_):
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

    # ======== fisher merge ========
    _fisher_merged_model, best_fisher_result, best_fisher_coeff = merge(
        models,
        ds,
        metric,
        coefficient_type="grid",
        method="fisher",
        num_coefficients=FLAGS.num_coefficients,
    )

    print(f"Best fisher result: {best_fisher_result}")
    print(f"Best fisher coefficients: {best_fisher_coeff}")

    # ======== isotropic merge ========
    _isotropic_merged_model, best_result, best_coeff = merge(
        models,
        ds,
        metric,
        coefficient_type="grid",
        method="isotropic",
        normalize=False,
        num_coefficients=FLAGS.num_coefficients,
    )

    print(f"Best isotropic result: {best_result}")
    print(f"Best isotropic coefficients: {best_coeff}")


if __name__ == "__main__":
    app.run(main)
