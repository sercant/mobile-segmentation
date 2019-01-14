import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv('SLIM_PATH'))
