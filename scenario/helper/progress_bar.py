import pandas as pd
import tqdm

class ProgressBar:

    def __init__(self, total, unit, desc, csv_path):
        self.pbar = tqdm.tqdm(total=total, unit=unit, desc=desc)
        self.unit = unit
        self.csv_data = {}
        self.csv_path = csv_path
        self.global_step = 0

    def _update_step(self):
        self.pbar.update(1)
        self.global_step += 1

    def _append_csv_data(self, **kwargs):
        if self.unit not in self.csv_data:
            self.csv_data[self.unit] = []
        self.csv_data[self.unit].append(self.global_step)
        for k in kwargs:
            if k not in self.csv_data:
                self.csv_data[k] = []
            self.csv_data[k].append(kwargs[k])

    def _update_description(self, **kwargs):
        _kwargs = {}
        for k, v in kwargs.items():
            if any(t in k for t in ('loss', 'psnr', 'snr', 'acc')):
                _kwargs[k] = f'{v:0.6f}' if isinstance(v, float) else v
            else:
                _kwargs[k] = v
        self.pbar.set_postfix(**_kwargs)

    def _display(self):
        self.pbar.display()

    def step(self, **kwargs):
        self._update_description(**kwargs)
        self._append_csv_data(**kwargs)
        self._update_step()
        self._display()

    def to_csv(self):
        df = pd.DataFrame(self.csv_data)
        df.to_csv(self.csv_path, index=None)

def create(total, unit, desc, csv_path):
    return ProgressBar(total, unit, desc, csv_path)

if __name__ == '__main__':
    import time
    pbar = ProgressBar(total=10, unit='iteration', desc='demo',
                       csv_path='../../data/csv/test.csv')
    for i in range(10):
        pbar.step(i=i, i2=i**2)
        time.sleep(1)
    pbar.to_csv()
