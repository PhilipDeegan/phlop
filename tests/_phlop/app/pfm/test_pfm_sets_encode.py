# tests/_phlop/app/pfm/test_pfm_sets_encode.py
#
# Every libpfm event in every PMU's sets encodes, via LIBPFM_FORCE_PMU: needs
# libpfm (./sh/setup_pfm.sh) but no PMU or counter access, so it runs on VMs.
# A libpfm build knows its own architecture's models only, others are skipped.
# Encoding is not counting: it checks event names, not what they count.

import json
import os
import subprocess
import sys
import unittest

from phlop.app.pfm import check_events as ce
from phlop.app.pfm.sets import SETS

ENCODE = """
import json, sys
from phlop.app.pfm import check_events as ce
from phlop.app.pfm.sets import SETS
if sys.argv[1] not in ce.detected_pmus():
    print(json.dumps(None))
    sys.exit(0)
events = sorted({
    spec for s in SETS[sys.argv[1]] for spec in s.events.values()
    if not spec.startswith(("raw:", "perf:"))
})
try:
    print(json.dumps({"encoded": len(ce.encode(events))}))
except ValueError as e:
    print(json.dumps({"error": str(e)}))
"""


def encode_forced(pmu):
    env = {**os.environ, ce.FORCE_PMU_ENV: pmu}
    out = subprocess.run(
        [sys.executable, "-c", ENCODE, pmu],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.splitlines()[-1])


class PfmSetsEncodeTest(unittest.TestCase):
    def test_sets_encode_for_forced_pmus(self):
        if not (ce.pfm_examples_dir() / "check_events").exists():
            self.skipTest("libpfm not built, see ./sh/setup_pfm.sh")
        checked = []
        for pmu in SETS:
            with self.subTest(pmu=pmu):
                result = encode_forced(pmu)
                if result is None:
                    print(f"{pmu}: not in this libpfm build, skipped")
                    continue
                self.assertNotIn("error", result, f"{pmu}: {result.get('error')}")
                print(f"{pmu}: {result['encoded']} events encode")
                checked.append(pmu)
        self.assertTrue(checked, "no PMU with sets is in this libpfm build")


if __name__ == "__main__":
    unittest.main()
