import sys
import unittest

from test_transcribe import *
import test_transcribe

if __name__ == '__main__':

    # Handle several ways of generating expected outputs
    if "--long" in sys.argv:
        test_transcribe.SKIP_LONG_TEST_IF_CPU = False
        sys.argv.remove("--long")
    if "--generate" in sys.argv:
        test_transcribe.FAIL_IF_REFERENCE_NOT_FOUND = False
        sys.argv.remove("--generate")
    if "--generate_device" in sys.argv:
        test_transcribe.GENERATE_DEVICE_DEPENDENT = True
        test_transcribe.FAIL_IF_REFERENCE_NOT_FOUND = False
        sys.argv.remove("--generate_device")
    if "--generate_new" in sys.argv:
        test_transcribe.GENERATE_NEW_ONLY = True
        test_transcribe.FAIL_IF_REFERENCE_NOT_FOUND = False
        sys.argv.remove("--generate_new")
    if "--generate_all" in sys.argv:
        test_transcribe.GENERATE_ALL = True
        test_transcribe.FAIL_IF_REFERENCE_NOT_FOUND = False
        sys.argv.remove("--generate_all")

    # Pass options to whisper_timestamped CLI
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg not in [
            "-h", "--help",
            "-v", "--verbose",
            "--locals",
            "-q", "--quiet",
            "-f", "--failfast",
            "-c", "--catch",
            "-b", "--buffer",
            "-k",
        ] and (i==0 or args[i-1] not in ["-k"]) and (arg.startswith("-") or (i>0 and args[i-1].startswith("-"))):
            test_transcribe.CMD_OPTIONS.append(arg)
            sys.argv.remove(arg)

    unittest.main()
