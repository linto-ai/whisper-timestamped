import sys
import unittest

from test_transcribe import TestTranscribe
import test_transcribe

if __name__ == '__main__':

    if "--generate_new" in sys.argv:
        test_transcribe.GENERATE_NEW_ONLY = True
        sys.argv.remove("--generate_new")
    if "--regenerate_all" in sys.argv:
        test_transcribe.REGENERATE_ALL = True
        sys.argv.remove("--regenerate_all")

    unittest.main()
