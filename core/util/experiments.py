import base64
import hashlib
import os
from datetime import datetime


def setup_experiment(experiment_dir):
    time_now = datetime.now()
    hashfun = hashlib.sha1()
    hashfun.update(b"{time_now}")
    tag = base64.b64encode(hashfun.digest()).decode()[:10] + "_" + time_now.strftime("%m%d_%I%M%S")
    print(f"Experiment tag is {tag}")
    output_dir = os.path.join(experiment_dir, tag)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    return output_dir, tag
