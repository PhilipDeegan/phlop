#
#
#
#
#

import sys
import time

if __name__ == "__main__":
    t = sys.argv[1]
    if not t:
        t = 1
    time.sleep(int(t))
