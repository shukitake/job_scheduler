import contextlib
import sys
from typing import Optional

import joblib
from tqdm.auto import tqdm


class LoggerUtil:
    @staticmethod
    def get_logger(name: str):
        from loguru import logger

        logger.remove()
        custom_format = "<green>[{extra[name]} {time:YYYY-MM-DD HH:mm:ss}]</green> <level>{level} {message}</level>"
        logger.add(sys.stdout, colorize=True, format=custom_format)
        logger = logger.bind(name=name)
        return logger

    @contextlib.contextmanager
    def tqdm_joblib(total: Optional[int] = None, **kwargs):
        pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                pbar.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

        try:
            yield pbar
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            pbar.close()
