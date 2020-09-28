import os
from transformers.data.processors.utils import DataProcessor, InputExample

class WikiNLIProcessor(DataProcessor):
    """Processor for the Wiki NLI data set.""" 

    def get_train_examples(self, data_dir, num_train_examples):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", num_examples=num_train_examples)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["parent", "child", "neutral"]

    def _create_examples(self, lines, set_type, num_examples=24000):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i >= num_examples:
                break
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=str(i), text_a=text_a, text_b=text_b, label=label))
        return examples

class WikiNLIFourWayProcessor(WikiNLIProcessor):
    """Processor for the Wiki NLI data set.""" 

    def get_labels(self):
        """See base class."""
        return ["parent", "child", "neutral", "sibling"]

class WikiNLIBinaryProcessor(WikiNLIProcessor):
    """Processor for the Wiki NLI binary data set.""" 

    def get_labels(self):
        """See base class."""
        return ["entailment", "non-entailment"]

processors = {
    "wikinli": WikiNLIProcessor,
    "wikinlifourway": WikiNLIFourWayProcessor,
    "wikinlibinary": WikiNLIBinaryProcessor,
}

output_modes = {
    "wikinli": "classification",
    "wikinlifourway": "classification",
    "wikinlibinary": "classification",
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    if task_name in ["wikinli", "wikinlifourway", "wikinlibinary"]:
        return {"acc": simple_accuracy(preds, labels)}
