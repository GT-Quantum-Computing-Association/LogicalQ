import argparse

from LogicalQ.tests.TestQEC import TestAllQEC
from LogicalQ.tests.TestLogicalGates import TestAllGates

from LogicalQ.Library.QECCs import implemented_codes

TestAllQEC(implemented_codes)

TestAllGates(implemented_codes)

