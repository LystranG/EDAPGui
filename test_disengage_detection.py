"""测试超巡脱离提示的 OCR 判定和触发门控。"""

import unittest
import ast
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
from unittest.mock import Mock

import cv2
import numpy as np

from DisengageDetection import evaluate_disengage_text, normalize_ocr_text, can_trigger_disengage


def load_sc_disengage_ocr():
    """提取单个方法测试，避免导入 ED_AP 时触发日志文件重命名。"""
    source = Path(__file__).with_name('ED_AP.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_clear_disengage_overlay', '_show_disengage_overlay',
                                 'sc_disengage_ocr', 'stop_sco_monitoring'}]
    namespace = {
        'cv2': cv2,
        'time': __import__('time'),
        'GuiFocusNoFocus': 0,
        'Flags2FsdScoActive': 1,
        'can_trigger_disengage': can_trigger_disengage,
        'evaluate_disengage_text': evaluate_disengage_text,
        'logger': Mock(),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'ED_AP.py', 'exec'), namespace)
    return (namespace['_clear_disengage_overlay'], namespace['_show_disengage_overlay'],
            namespace['sc_disengage_ocr'], namespace['stop_sco_monitoring'])


SC_CLEAR_OVERLAY, SC_SHOW_OVERLAY, SC_DISENGAGE_OCR, STOP_SCO_MONITORING = load_sc_disengage_ocr()


class DisengageDetectionTests(unittest.TestCase):
    def test_normalizes_ocr_fragments(self):
        self.assertEqual(normalize_ocr_text(['C', 'PRESS [J] TO DISE']), 'CPRESSJ TODISE'.replace(' ', ''))

    def test_accepts_complete_prompt_even_with_keycap(self):
        result = evaluate_disengage_text(['PRESS [J] TO DISENGAGE'], 0.1, 0.35)
        self.assertTrue(result.trigger)
        self.assertEqual(result.reason, 'phrase')

    def test_accepts_unavoidably_truncated_prompt(self):
        result = evaluate_disengage_text(['C', 'PRESS [J] TO DISE'], 0.2, 0.35)
        self.assertTrue(result.trigger)
        self.assertEqual(result.reason, 'phrase')

    def test_similarity_fallback_accepts_partial_but_strong_result(self):
        result = evaluate_disengage_text(['PRESS [J] TO DIS'], 0.36, 0.35)
        self.assertTrue(result.trigger)
        self.assertEqual(result.reason, 'similarity')

    def test_random_ocr_text_does_not_trigger(self):
        result = evaluate_disengage_text(['C', '4', 'E5'], 0.9, 0.35)
        self.assertFalse(result.trigger)
        self.assertEqual(result.reason, 'no-match')

        result = evaluate_disengage_text(['PRESSURE', 'DISTANCE'], 0.9, 0.35)
        self.assertFalse(result.trigger)

        result = evaluate_disengage_text(['PRESSURE', 'TO', 'DISEASE'], 0.9, 0.35)
        self.assertFalse(result.trigger)

        result = evaluate_disengage_text(['SUPPRESS', 'AUTO', 'DISENGAGED'], 0.9, 0.35)
        self.assertFalse(result.trigger)

    def test_non_english_locale_uses_locale_similarity(self):
        result = evaluate_disengage_text(['Нажмите чтобы остановить'], 0.9, 0.35,
                                         'Нажмите чтобы остановить')
        self.assertTrue(result.trigger)
        self.assertEqual(result.reason, 'similarity')

    def test_empty_locale_prompt_does_not_allow_arbitrary_similarity(self):
        result = evaluate_disengage_text(['RANDOM TEXT'], 1.0, 0.35, '?')
        self.assertFalse(result.trigger)

    def test_disengage_requires_cockpit_and_inactive_sco(self):
        self.assertTrue(can_trigger_disengage(gui_focus=0, sco_active=False))
        self.assertFalse(can_trigger_disengage(gui_focus=3, sco_active=False))
        self.assertFalse(can_trigger_disengage(gui_focus=0, sco_active=True))

    def test_debug_status_explains_gate(self):
        result = evaluate_disengage_text(['PRESS [J] TO DISE'], 0.2, 0.35)
        self.assertIn('trigger', result.status_text)
        self.assertIn('sim=0.200', result.status_text)


class DisengageMethodTests(unittest.TestCase):
    def call_method(self, ocr_text, similarity, gui_focus=0, sco_active=False, debug_overlay=True):
        keys = Mock()
        overlay = Mock()
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        ap = SimpleNamespace(
            status=SimpleNamespace(get_gui_focus=lambda: gui_focus),
            sc_sco_is_active=sco_active,
            _sc_disengage_cancel_epoch=0,
            debug_overlay=debug_overlay,
            cv_view=False,
            scr=SimpleNamespace(get_screen_region=lambda _: image),
            ocr=SimpleNamespace(image_simple_ocr=lambda *_: ocr_text,
                                 string_similarity=lambda *_: similarity),
            locale={'PRESS_TO_DISENGAGE_MSG': 'PRESS TO DISENGAGE'},
            keys=keys,
            ap_ckb=Mock(),
            overlay=overlay,
        )
        ap._clear_disengage_overlay = MethodType(SC_CLEAR_OVERLAY, ap)
        ap._show_disengage_overlay = MethodType(SC_SHOW_OVERLAY, ap)
        scr_reg = SimpleNamespace(
            reg={'disengage': {'rect': [1, 2, 10, 12]}},
            capture_region_filtered=lambda *_: np.zeros((20, 20), dtype=np.uint8),
        )
        result = SC_DISENGAGE_OCR(ap, scr_reg)
        labels = [call.args[1] for call in overlay.overlay_floating_text.call_args_list]
        return result, keys, labels

    def test_method_sends_j_for_truncated_prompt(self):
        result, keys, labels = self.call_method(['C', 'PRESS [J] TO DISE'], 0.2)
        self.assertTrue(result)
        keys.send.assert_called_once_with('HyperSuperCombination')
        self.assertTrue(any('SEND J' in label for label in labels))

    def test_method_waits_for_random_ocr_even_with_high_similarity(self):
        result, keys, labels = self.call_method(['C', '4', 'E5'], 0.9)
        self.assertFalse(result)
        keys.send.assert_not_called()
        self.assertTrue(any('WAIT' in label and 'no-match' in label for label in labels))

    def test_method_waits_when_sco_is_active(self):
        result, keys, labels = self.call_method(['PRESS [J] TO DISENGAGE'], 0.9, sco_active=True)
        self.assertFalse(result)
        keys.send.assert_not_called()
        self.assertTrue(any('sco=1 WAIT' in label for label in labels))

    def test_method_explains_non_cockpit_focus(self):
        result, keys, labels = self.call_method(['PRESS [J] TO DISENGAGE'], 0.9, gui_focus=3)
        self.assertFalse(result)
        keys.send.assert_not_called()
        self.assertTrue(any('gui=3' in label and 'WAIT (focus)' in label for label in labels))

    def test_method_rechecks_gate_before_sending(self):
        focus_values = iter((0, 3))
        ap_status = SimpleNamespace(get_gui_focus=lambda: next(focus_values))
        keys = Mock()
        ap = SimpleNamespace(
            status=ap_status, sc_sco_is_active=False, _sc_disengage_cancel_epoch=0,
            debug_overlay=False, cv_view=False,
            scr=SimpleNamespace(get_screen_region=lambda _: np.zeros((20, 20, 3), dtype=np.uint8)),
            ocr=SimpleNamespace(image_simple_ocr=lambda *_: ['PRESS [J] TO DISENGAGE'],
                                 string_similarity=lambda *_: .9),
            locale={'PRESS_TO_DISENGAGE_MSG': 'PRESS TO DISENGAGE'}, keys=keys,
            ap_ckb=Mock(), overlay=Mock(),
        )
        scr_reg = SimpleNamespace(reg={'disengage': {'rect': [1, 2, 10, 12]}},
                                  capture_region_filtered=lambda *_: np.zeros((20, 20), dtype=np.uint8))
        self.assertFalse(SC_DISENGAGE_OCR(ap, scr_reg))
        keys.send.assert_not_called()

    def test_monitor_shutdown_preserves_disengage_latch_until_docking(self):
        """监控线程因离开 SC 停止时，不能抢先清除主循环需要的脱离锁存。"""
        overlay = Mock()
        ap = SimpleNamespace(
            _sc_sco_active_loop_enable=True,
            _sc_disengage_cancel_epoch=0,
            _sc_disengage_active=True,
            overlay=overlay,
        )
        ap._clear_disengage_overlay = MethodType(SC_CLEAR_OVERLAY, ap)

        STOP_SCO_MONITORING(ap, clear_disengage=False)
        self.assertTrue(ap._sc_disengage_active)

        STOP_SCO_MONITORING(ap)
        self.assertFalse(ap._sc_disengage_active)

    def test_method_rechecks_sco_before_sending(self):
        keys = Mock()
        ap = SimpleNamespace(
            status=SimpleNamespace(get_gui_focus=lambda: 0,
                                   get_flag2=lambda _: True),
            sc_sco_is_active=False, _sc_disengage_cancel_epoch=0,
            debug_overlay=False, cv_view=False,
            scr=SimpleNamespace(get_screen_region=lambda _: np.zeros((20, 20, 3), dtype=np.uint8)),
            ocr=SimpleNamespace(image_simple_ocr=lambda *_: ['PRESS [J] TO DISENGAGE'],
                                 string_similarity=lambda *_: .9),
            locale={'PRESS_TO_DISENGAGE_MSG': 'PRESS TO DISENGAGE'}, keys=keys,
            ap_ckb=Mock(), overlay=Mock(),
        )
        scr_reg = SimpleNamespace(reg={'disengage': {'rect': [1, 2, 10, 12]}},
                                  capture_region_filtered=lambda *_: np.zeros((20, 20), dtype=np.uint8))
        self.assertFalse(SC_DISENGAGE_OCR(ap, scr_reg))
        keys.send.assert_not_called()

    def test_method_does_not_send_after_monitor_stop_during_ocr(self):
        keys = Mock()
        ap = SimpleNamespace(
            status=SimpleNamespace(get_gui_focus=lambda: 0), sc_sco_is_active=False,
            _sc_disengage_cancel_epoch=0, debug_overlay=False, cv_view=False,
            scr=SimpleNamespace(get_screen_region=lambda _: np.zeros((20, 20, 3), dtype=np.uint8)),
            locale={'PRESS_TO_DISENGAGE_MSG': 'PRESS TO DISENGAGE'}, keys=keys,
            ap_ckb=Mock(), overlay=Mock(),
        )

        def ocr_then_stop(*_):
            ap._sc_disengage_cancel_epoch += 1
            return ['PRESS [J] TO DISENGAGE']

        ap.ocr = SimpleNamespace(image_simple_ocr=ocr_then_stop,
                                 string_similarity=lambda *_: .9)
        scr_reg = SimpleNamespace(reg={'disengage': {'rect': [1, 2, 10, 12]}},
                                  capture_region_filtered=lambda *_: np.zeros((20, 20), dtype=np.uint8))
        self.assertFalse(SC_DISENGAGE_OCR(ap, scr_reg))
        keys.send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
