"""离线回归罗盘方向，不初始化游戏窗口、OCR 或按键控制。"""

import ast
from copy import copy
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parent
REPOSITORY = Path(os.environ.get('EDAP_TEST_REPOSITORY', SOURCE))
sys.path.insert(0, str(REPOSITORY))
from MachineLearning import MachLearnMatch, ModelType
from Screen_Regions import Quad


def load_navigation_methods():
    # 只提取被测方法，避免导入主程序时启动桌面服务。
    tree = ast.parse((SOURCE / 'ED_AP.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EDAutopilot')
    names = {'get_nav_offset', 'get_compass_target_offset', 'compass_align'}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(cv2=cv2, math=math, atan=math.atan, degrees=math.degrees,
                     Quad=Quad, ModelType=ModelType, copy=copy, logger=Mock())
    try:
        from CompassDetection import detect_navpoint_shape
        namespace['detect_navpoint_shape'] = detect_navpoint_shape
    except ImportError:
        pass
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / 'ED_AP.py'), 'exec'), namespace)
    return namespace


METHODS = load_navigation_methods()


def match(name, score, bounds):
    return MachLearnMatch(name, score, Quad.from_rect(bounds))


def synthetic_marker(behind, radius=6, center=(60, 60)):
    image = np.full((120, 120, 3), (15, 25, 40), dtype=np.uint8)
    cv2.circle(image, (60, 60), 45, (0, 130, 230), 2)
    cv2.circle(image, center, radius, (190, 210, 110), 2 if behind else -1, cv2.LINE_AA)
    return image


class CompassDirectionTests(unittest.TestCase):
    def nav_offset(self, image, matches, overlay=False):
        reg = SimpleNamespace(capture_region_percent=lambda *_: image,
                              reg={'compass': {'rect': [0, 0, image.shape[1], image.shape[0]]}},
                              compass_match_thresh=.5, navpoint_match_thresh=.8)
        ap = SimpleNamespace(scr=None, mach_learn=SimpleNamespace(model_predict=lambda *_: matches),
                             debug_images=False, debug_overlay=overlay, cv_view=False, overlay=Mock())
        return METHODS['get_nav_offset'](ap, reg), ap

    def test_hollow_overrides_front_model_class(self):
        result, _ = self.nav_offset(synthetic_marker(True), [match('compass', .9, [15, 15, 105, 105]), match('navpoint', .32, [52, 52, 68, 68])])
        self.assertIsNotNone(result)
        self.assertEqual(result['z'], -1)
        self.assertGreater(abs(result['pit']), 170)

    def test_solid_overrides_behind_model_class(self):
        result, _ = self.nav_offset(synthetic_marker(False), [match('compass', .9, [15, 15, 105, 105]), match('navpoint-behind', .4, [52, 52, 68, 68])])
        self.assertEqual(result['z'], 1)
        self.assertLess(abs(result['pit']), 3)

    def test_shape_recovers_model_missing_marker(self):
        for behind in (True, False):
            with self.subTest(behind=behind):
                result, _ = self.nav_offset(synthetic_marker(behind), [match('compass', .9, [15, 15, 105, 105])])
                self.assertIsNotNone(result)
                self.assertEqual(result['z'], -1 if behind else 1)

    def test_no_marker_remains_unknown(self):
        result, _ = self.nav_offset(np.zeros((120, 120, 3), np.uint8), [match('compass', .9, [15, 15, 105, 105])])
        self.assertIsNone(result)

    def test_unsupported_color_keeps_model_direction(self):
        result, _ = self.nav_offset(np.zeros((120, 120, 3), np.uint8), [match('compass', .9, [15, 15, 105, 105]), match('navpoint-behind', .8, [52, 52, 68, 68])])
        self.assertEqual(result['z'], -1)

    def test_model_uses_strongest_match_inside_compass(self):
        result, _ = self.nav_offset(np.zeros((160, 160, 3), np.uint8), [match('compass', .95, [15, 15, 105, 105]), match('compass', .3, [120, 120, 155, 155]), match('navpoint-behind', .8, [52, 52, 68, 68]), match('navpoint', .99, [130, 130, 140, 140])])
        self.assertEqual(result['z'], -1)

    def test_model_marker_can_touch_compass_edge(self):
        result, _ = self.nav_offset(np.zeros((120, 120, 3), np.uint8), [match('compass', .9, [15, 15, 105, 105]), match('navpoint-behind', .99, [52, 6, 68, 22])])
        self.assertIsNotNone(result)
        self.assertEqual(result['z'], -1)

    def test_boundary_marker_uses_ninety_degree_probe(self):
        result, _ = self.nav_offset(np.zeros((120, 120, 3), np.uint8),
                                    [match('compass', .9, [15, 15, 105, 105]),
                                     match('navpoint', .8, [15, 52, 31, 68])])
        self.assertTrue(result['boundary'])
        self.assertGreaterEqual(abs(result['yaw']), 90)

    def test_actual_screenshots_override_wrong_model_class(self):
        fixtures = [('front.png', False), ('behind.png', True), ('behind_small.png', True)]
        for filename, behind in fixtures:
            with self.subTest(filename=filename):
                image = cv2.imread(str(SOURCE / 'test' / 'compass_regression' / filename))
                self.assertIsNotNone(image)
                for scale in (.75, 1, 1.5, 2):
                    resized = cv2.resize(image, None, fx=scale, fy=scale)
                    h, w = resized.shape[:2]
                    result, _ = self.nav_offset(resized, [match('compass', .9, [0, 0, w, h]), match('navpoint' if behind else 'navpoint-behind', .32, [w/2-5, h/2-5, w/2+5, h/2+5])])
                    self.assertIsNotNone(result)
                    self.assertEqual(result['z'], -1 if behind else 1, (filename, scale))

    def test_boundary_marker_recovers_compass_when_model_misses_ring(self):
        """导航点贴近圆环边界时，不能因罗盘本体低置信度而进入无条件 roll。"""
        template = cv2.imread(str(SOURCE / 'templates' / 'compass.png'), cv2.IMREAD_GRAYSCALE)
        h, w = template.shape[:2]
        # 用模板构造最小边界场景：罗盘完整可匹配，导航点框贴在圆环上边缘。
        image = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        # 模型仍能给出低分导航点，但漏掉低分罗盘本体；生产代码应使用模板恢复罗盘框。
        nav = match('navpoint', .115, [34, 0, 45, 10])
        reg = SimpleNamespace(
            capture_region_percent=lambda *_: cv2.cvtColor(image, cv2.COLOR_BGR2BGRA),
            reg={'compass': {'rect': [0, 0, image.shape[1], image.shape[0]]}},
            compass_match_thresh=.5, navpoint_match_thresh=.8,
            templates=SimpleNamespace(template={'compass': {'image': template, 'width': w, 'height': h}}))
        ap = SimpleNamespace(scr=None, mach_learn=SimpleNamespace(model_predict=lambda *_: [nav]),
                             debug_images=False, debug_overlay=False, cv_view=False, overlay=Mock())
        result = METHODS['get_nav_offset'](ap, reg)
        self.assertIsNotNone(result)
        self.assertLess(abs(result['roll']), 20)

    def test_template_fallback_handles_empty_model_result(self):
        """模型完全无结果时，模板识别罗盘后仍应能从图像定位导航点。"""
        template = cv2.imread(str(SOURCE / 'templates' / 'compass.png'), cv2.IMREAD_GRAYSCALE)
        h, w = template.shape[:2]
        image = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        cv2.circle(image, (w // 2, h // 2), max(5, min(w, h) // 12), (190, 210, 110), -1)
        reg = SimpleNamespace(
            capture_region_percent=lambda *_: cv2.cvtColor(image, cv2.COLOR_BGR2BGRA),
            reg={'compass': {'rect': [0, 0, w, h]}},
            compass_match_thresh=.5, navpoint_match_thresh=.8,
            templates=SimpleNamespace(template={'compass': {'image': template, 'width': w, 'height': h}}))
        ap = SimpleNamespace(scr=None, mach_learn=SimpleNamespace(model_predict=lambda *_: None),
                             debug_images=False, debug_overlay=False, cv_view=False, overlay=Mock())
        result = METHODS['get_nav_offset'](ap, reg)
        self.assertIsNotNone(result)
        self.assertEqual(result['z'], 1)

    def test_behind_compass_takes_precedence_over_front_target(self):
        ap = SimpleNamespace(scrReg=None, ap_ckb=Mock(),
                             get_nav_offset=lambda _: dict(z=-1, roll=0, pit=160, yaw=179),
                             get_target_offset=lambda _: dict(roll=0, pit=0, yaw=0, occ=False))
        result = METHODS['get_compass_target_offset'](ap)
        self.assertTrue(result['tar_behind'])
        self.assertTrue(result['used_nav'])
        self.assertEqual(result['pit'], 160)

    def test_front_target_keeps_precision_priority(self):
        ap = SimpleNamespace(scrReg=None, ap_ckb=Mock(),
                             get_nav_offset=lambda _: dict(z=1, roll=0, pit=2, yaw=2),
                             get_target_offset=lambda _: dict(roll=0, pit=1, yaw=1, occ=False))
        result = METHODS['get_compass_target_offset'](ap)
        self.assertTrue(result['used_tar'])
        self.assertEqual(result['pit'], 1)

    def test_overlay_keeps_model_scores_and_shows_final_direction(self):
        _, ap = self.nav_offset(synthetic_marker(True), [match('compass', .9, [15, 15, 105, 105]), match('navpoint', .32, [52, 52, 68, 68])], overlay=True)
        labels = {call.args[0]: call.args[1] for call in ap.overlay.overlay_floating_text.call_args_list}
        self.assertIn('0.00', labels['nav_beh'])
        self.assertIn('behind', labels['nav_beh'])
        self.assertIn('shape', labels['nav_beh'])
        self.assertNotIn('> 0.8', labels['nav'])


class ShapeValidationTests(unittest.TestCase):
    def classify(self, image):
        from CompassDetection import detect_navpoint_shape
        return detect_navpoint_shape(image, Quad.from_rect([15, 15, 105, 105]))

    def test_white_highlight_is_not_a_hollow_center(self):
        image = synthetic_marker(True)
        cv2.circle(image, (60, 60), 4, (245, 245, 245), -1)
        result = self.classify(image)
        self.assertTrue(result is None or not result.behind)

    def test_partial_arc_does_not_imply_hollow(self):
        image = synthetic_marker(True)
        image[50:72, 60:72] = (15, 25, 40)
        self.assertIsNone(self.classify(image))

    def test_multiple_markers_are_ambiguous(self):
        image = synthetic_marker(False, center=(40, 60))
        cv2.circle(image, (80, 60), 6, (190, 210, 110), 2, cv2.LINE_AA)
        self.assertIsNone(self.classify(image))

    def test_green_overlay_is_not_a_marker(self):
        image = np.zeros((120, 120, 3), np.uint8)
        cv2.circle(image, (60, 60), 6, (0, 255, 0), 2)
        self.assertIsNone(self.classify(image))

    def test_blurred_markers_keep_direction(self):
        for behind in (False, True):
            for sigma in (.5, 1):
                with self.subTest(behind=behind, sigma=sigma):
                    result = self.classify(cv2.GaussianBlur(synthetic_marker(behind), (3, 3), sigma))
                    self.assertIsNotNone(result)
                    self.assertEqual(result.behind, behind)

    def test_marker_near_compass_rim(self):
        for behind in (False, True):
            with self.subTest(behind=behind):
                result = self.classify(synthetic_marker(behind, center=(60, 17)))
                self.assertIsNotNone(result)
                self.assertEqual(result.behind, behind)

    def test_noise_does_not_change_direction(self):
        noise = np.random.default_rng(42).normal(0, 3, (120, 120, 3))
        for behind in (False, True):
            with self.subTest(behind=behind):
                noisy = np.clip(synthetic_marker(behind).astype(float) + noise, 0, 255).astype(np.uint8)
                result = self.classify(noisy)
                self.assertIsNotNone(result)
                self.assertEqual(result.behind, behind)


if __name__ == '__main__':
    unittest.main()
