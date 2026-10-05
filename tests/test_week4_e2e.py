"""
Automated Pytest wrapper for Week 4 End-to-End Scenarios.
==========================================================
Executes all 9 required end-to-end technical scenarios in pytest.
"""

import pytest
from tests.week4_e2e_scenario_runner import run_week4_scenarios


@pytest.fixture(scope="module")
def scenario_results():
    """Run Week 4 scenarios once for the test module."""
    return run_week4_scenarios()


@pytest.mark.parametrize("scenario_idx", range(9))
def test_week4_scenario_passes(scenario_results, scenario_idx):
    """Verify each of the 9 technical scenarios achieves PASS status."""
    scenario = scenario_results[scenario_idx]
    assert scenario["Pass/Fail"] == "PASS", (
        f"Scenario '{scenario['Test']}' failed! "
        f"Expected: {scenario['Expected Result']} | Actual: {scenario['Actual Result']}"
    )
