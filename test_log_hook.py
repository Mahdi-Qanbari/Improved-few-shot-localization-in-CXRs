from mmdet_custom.hooks.log_classwise_map_hook import LogClasswiseMAPHook

# Dummy runner to simulate MMEngine's runner object
class DummyLogger:
    def info(self, msg): print(msg)
    def warning(self, msg): print("[WARNING]", msg)

class DummyRunner:
    def __init__(self, epoch):
        self.epoch = epoch
        self.logger = DummyLogger()

# Dummy metrics like what CocoMetric returns with classwise=True
fake_metrics = {
    'metrics': True,
    'results_per_category': [
        ('Pneumonia', 0.823),
        ('Atelectasis', 0.512),
        ('Nodule', 0.731),
        ('Mass', None),  # Simulate missing value
    ]
}

# Create hook instance
hook = LogClasswiseMAPHook(csv_path='work_dirs/test_classwise_map.csv')

# Call the logging method manually
hook._log_to_csv(DummyRunner(epoch=4), fake_metrics)

