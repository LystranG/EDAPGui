"""空间站 waypoint 流程的失败传播回归测试。"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase


SOURCE = Path(__file__).resolve().parent


def load_method():
    tree = ast.parse((SOURCE / 'ED_AP.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'supercruise_to_station')
    namespace = {
        'sleep': lambda *_args, **_kwargs: None,
        'FlagsDocked': object(),
        'FlagsLanded': object(),
    }
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace['supercruise_to_station']


def load_sc_assist():
    tree = ast.parse((SOURCE / 'ED_AP.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'sc_assist')
    namespace = {'logger': SimpleNamespace(debug=lambda *_args, **_kwargs: None),
                 'sleep': lambda *_args, **_kwargs: None}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace['sc_assist']


SUPERCRUISE_TO_STATION = load_method()
SC_ASSIST = load_sc_assist()


class StationFlowTests(TestCase):
    def make_ap(self, assist_result):
        status = SimpleNamespace(get_flag=lambda *_: False)
        ap = SimpleNamespace(
            status=status,
            update_ap_status=lambda *_: None,
            waypoint_undock_seq=lambda: None,
            sc_engage=lambda *_: None,
            have_destination=lambda *_: True,
            ap_ckb=lambda *_: None,
            config={'CompassReacquireTries': 2},
            sc_assist=lambda *_: assist_result,
        )
        return ap

    def test_sc_failure_is_not_reported_as_station_success(self):
        ap = self.make_ap(False)
        self.assertFalse(SUPERCRUISE_TO_STATION(ap, None, 'TEST STATION'))

    def test_sc_success_is_reported_as_station_success(self):
        ap = self.make_ap(True)
        self.assertTrue(SUPERCRUISE_TO_STATION(ap, None, 'TEST STATION'))

    def test_station_flow_retries_transient_compass_loss(self):
        results = iter((False, False, True))
        ap = self.make_ap(True)
        ap.have_destination = lambda *_: next(results)
        self.assertTrue(SUPERCRUISE_TO_STATION(ap, None, 'TEST STATION'))

    def test_sc_assist_returns_false_when_compass_is_missing(self):
        ap = SimpleNamespace(
            ship_control=SimpleNamespace(goto_cockpit_view=lambda: None,
                                         roll_clockwise_anticlockwise=lambda *_: None),
            have_destination=lambda *_: False,
            ap_ckb=lambda *_: None,
            config={'CompassReacquireTries': 2},
        )
        self.assertFalse(SC_ASSIST(ap, None))


if __name__ == '__main__':
    import unittest
    unittest.main()
