import ast, os, sys, unittest
from unittest.mock import MagicMock
for mod in ["torch", "torch.nn", "torch.nn.functional", "sklearn", "sklearn.cluster", "sklearn.metrics"]:
    if mod not in sys.modules: sys.modules[mod] = MagicMock()

class TestUnsupervisedFirewall(unittest.TestCase):
    def setUp(self):
        self.training_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline", "training.py"))
        with open(self.training_file, "r") as f: self.source_code = f.read()
        self.tree = ast.parse(self.source_code)
    def test_signatures(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in ["train_model", "evaluate_model"]:
                for a in node.args.args:
                    if "label" in a.arg.lower() or "ground_truth" in a.arg.lower(): self.fail()
    def test_no_supervised_metrics(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_model":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        fname = getattr(sub.func, "id", getattr(sub.func, "attr", ""))
                        if fname in ["adjusted_rand_score", "normalized_mutual_info_score"]: self.fail()
    def test_runtime_rejection(self):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from pipeline.training import train_model
        class BadArgs: true_labels = [0, 1]
        with self.assertRaises(PermissionError): train_model(MagicMock(), MagicMock(), BadArgs())
    def test_quarantined_posthoc(self):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from pipeline.training import evaluate_posthoc_quarantined
        res = evaluate_posthoc_quarantined([0, 1], [0, 1])
        self.assertTrue(res["quarantined_reporting_only"])
if __name__ == "__main__": unittest.main()
