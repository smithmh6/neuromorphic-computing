"""
Run snntoolbox.
"""

import os
import shutil
from snntoolbox.bin.run import main

if __name__ == "__main__":

    ##try:
    ##    os.remove("./models/vgg16/jaffe_nofear_run01/model_parsed.h5")
    ##except:
    ##    pass
    ##try:
    ##    os.remove("./models/vgg16/jaffe_nofear_run01/model_INI.h5")
    ##except:
    ##    pass
    ##try:
    ##    shutil.rmtree('./models/vgg16/jaffe_nofear_run01/log/gui/snn01')
    ##except:
    ##    pass

    main("./out/run_198/config.ini")

