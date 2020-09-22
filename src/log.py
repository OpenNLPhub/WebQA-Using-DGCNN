import logging
from datetime import datetime
import os


logger =  logging.getLogger('WenQA_DGCNN')
logger.setLevel(logging.INFO)

fmt = '[%(asctime)s]-[%(levelname)s]-%(message)s'
formatter=logging.Formatter(fmt)

handle_stream =  logging.StreamHandler()
handle_stream.setFormatter(formatter)
handle_stream.setLevel(logging.INFO)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cwd = os.getcwd()
log_path = os.path.join(cwd,'log',now+'.log')
handle_file = logging.FileHandler(filename = log_path,encoding = 'utf-8',mode = 'w')
handle_file.setFormatter(formatter)
handle_file.setLevel(logging.INFO)

logger.addHandler(handle_file)
logger.addHandler(handle_stream)
