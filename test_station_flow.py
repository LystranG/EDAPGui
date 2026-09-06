"""空间站 waypoint 流程的失败传播回归测试。"""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock


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
    class ScTargetAlignReturn(Enum):
        Lost = 1
        Found = 2
        Disengage = 3

    namespace = {
        'logger': SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        'sleep': lambda *_args, **_kwargs: None,
        'ScTargetAlignReturn': ScTargetAlignReturn,
        'FlagsSupercruise': object(),
        'Flags2GlideMode': object(),
        'FlagsDocked': object(),
        'FlagsLanded': object(),
    }
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace['sc_assist']


def load_sc_target_align():
    tree = ast.parse((SOURCE / 'ED_AP.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'sc_target_align')

    class ScTargetAlignReturn(Enum):
        Lost = 1
        Found = 2
        Disengage = 3

    namespace = {
        'logger': SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        'sleep': lambda *_args, **_kwargs: None,
        'ScTargetAlignReturn': ScTargetAlignReturn,
    }
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace['sc_target_align'], ScTargetAlignReturn


def load_sc_engage():
    tree = ast.parse((SOURCE / 'ED_AP.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'sc_engage')
    flags_supercruise = object()
    namespace = {'FlagsSupercruise': flags_supercruise}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace['sc_engage'], flags_supercruise


SUPERCRUISE_TO_STATION = load_method()
SC_ASSIST = load_sc_assist()
SC_TARGET_ALIGN, SC_TARGET_ALIGN_RETURN = load_sc_target_align()
SC_ENGAGE, FLAGS_SUPERCRUISE = load_sc_engage()


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

    def test_sc_assist_docks_after_destination_drop_without_disengage_ocr(self):
        ship = {
            'status': 'in_space',
            'SupercruiseDestinationDrop_type': 'TEST STATION',
            'has_adv_dock_comp': True,
            'has_std_dock_comp': False,
        }
        status = SimpleNamespace(
            get_flag=lambda *_: False,
            get_flag2=lambda *_: False,
        )
        ap = SimpleNamespace(
            ship_control=SimpleNamespace(goto_cockpit_view=Mock()),
            status=status,
            jn=SimpleNamespace(ship_state=lambda: ship),
            config={'CompassReacquireTries': 2},
            have_destination=Mock(return_value=True),
            sc_engage=Mock(),
            set_throttle_50=Mock(),
            set_throttle_100=Mock(),
            set_throttle_0=Mock(),
            compass_align=Mock(),
            sc_target_align=Mock(return_value=SC_TARGET_ALIGN_RETURN.Found),
            interdiction_check=Mock(return_value=False),
            dock=Mock(),
            ap_ckb=Mock(),
            vce=SimpleNamespace(say=Mock()),
            _sc_disengage_active=False,
            stop_sco_monitoring=Mock(),
            update_ap_status=Mock(),
        )

        result = SC_ASSIST(ap, None)

        self.assertTrue(result)
        ap.dock.assert_called_once_with()
        ap.stop_sco_monitoring.assert_called_once_with(clear_disengage=False)

    def test_new_supercruise_leg_clears_previous_disengage_latch(self):
        status = SimpleNamespace(get_flag=Mock(return_value=True))
        start_monitoring = Mock()
        ap = SimpleNamespace(
            status=status,
            _sc_disengage_active=True,
            _sc_disengage_cancel_epoch=0,
            _clear_disengage_overlay=Mock(),
            start_sco_monitoring=start_monitoring,
        )

        result = SC_ENGAGE(ap, False)

        self.assertTrue(result)
        self.assertFalse(ap._sc_disengage_active)
        self.assertEqual(ap._sc_disengage_cancel_epoch, 1)
        ap._clear_disengage_overlay.assert_called_once_with()
        start_monitoring.assert_called_once_with()

    def test_sc_target_align_tolerates_transient_target_loss(self):
        offsets = iter((
            {'pit': 3.0, 'yaw': 0.0, 'tar_behind': False, 'tar_occ': False, 'used_nav': False},
            None,
            None,
            {'pit': 0.25, 'yaw': 0.0, 'tar_behind': False, 'tar_occ': False, 'used_nav': False},
        ))
        ship_control = SimpleNamespace(pitch_up_down=Mock(), yaw_right_left=Mock())
        ap = SimpleNamespace(
            target_align_outer_lim=1.0,
            target_align_inner_lim=0.5,
            auto_tune_rpy=False,
            debug_overlay=False,
            _sc_disengage_active=False,
            ship_control=ship_control,
            get_compass_target_offset=lambda: next(offsets),
            ap_ckb=Mock(),
        )

        result = SC_TARGET_ALIGN(ap, None)

        self.assertEqual(result, SC_TARGET_ALIGN_RETURN.Found)
        ship_control.pitch_up_down.assert_called_once()

    def test_sc_target_align_still_fails_after_grace_retries(self):
        offsets = iter((
            {'pit': 3.0, 'yaw': 0.0, 'tar_behind': False, 'tar_occ': False, 'used_nav': False},
            None,
            None,
            None,
            None,
        ))
        ap = SimpleNamespace(
            target_align_outer_lim=1.0,
            target_align_inner_lim=0.5,
            auto_tune_rpy=False,
            debug_overlay=False,
            _sc_disengage_active=False,
            config={'TargetAlignLostRetries': 3, 'TargetAlignLostRetryDelay': 0.05},
            ship_control=SimpleNamespace(pitch_up_down=Mock(), yaw_right_left=Mock()),
            get_compass_target_offset=lambda: next(offsets),
            ap_ckb=Mock(),
        )

        result = SC_TARGET_ALIGN(ap, None)

        self.assertEqual(result, SC_TARGET_ALIGN_RETURN.Lost)
        ap.ap_ckb.assert_called_once_with('log', 'Target Align failed - lost target after recognition retries.')


if __name__ == '__main__':
    import unittest
    unittest.main()
