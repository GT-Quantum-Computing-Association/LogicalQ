from LogicalQ.tests.TestQEC import TestAllQEC
from LogicalQ.tests.TestLogicalGates import TestAllGates
from LogicalQ.tests.TestStateRepresentations import TestAllStateRepresentations

from LogicalQ.Library.QECCs import implemented_codes

TestAllQEC(implemented_codes)

TestAllGates(implemented_codes)

TestAllStateRepresentations(implemented_codes)

