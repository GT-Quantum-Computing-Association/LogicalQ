import argparse

from LogicalQ.tests.TestQEC import TestAllQEC
from LogicalQ.tests.TestLogicalGates import TestAllGates

from LogicalQ.Library.QECCs import implemented_codes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # QECC input
    parser.add_argument("--qecc-labels", default=None)

    # Test input
    parser.add_argument("--tests", default=None)

    args = parser.parse_args()

    # Parse QECC input
    if args.qecc_labels:
        qeccs = []
        for qecc_input in args.qecc_labels.split(";"):
            qecc = label_to_qecc(qecc_input)
            qeccs.append(qecc)
    else:
        qeccs = implemented_codes

    # Parse test input
    TestAllQEC(qeccs)
    TestAllGates(qeccs)

