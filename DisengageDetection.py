"""超巡脱离提示的 OCR 文本归一化、判定和状态门控。"""

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class DisengageDecision:
    """一次 OCR 结果的可解释判定。"""

    normalized_text: str
    similarity: float
    trigger: bool
    reason: str

    @property
    def status_text(self) -> str:
        action = 'trigger' if self.trigger else 'wait'
        return f'{action} reason={self.reason} sim={self.similarity:.3f}'


def normalize_ocr_text(ocr_textlist: Iterable[str] | str | None) -> str:
    """合并 OCR 片段并保留 Unicode 字母和数字，消除括号、空格及标点差异。"""
    if ocr_textlist is None:
        return ''
    if isinstance(ocr_textlist, str):
        text = ocr_textlist
    else:
        text = ' '.join(str(item) for item in ocr_textlist)
    return ''.join(char for char in text.upper() if char.isalnum())


def _ocr_tokens(ocr_textlist: Iterable[str] | str | None) -> list[str]:
    """按 OCR 词片段保留边界，避免 PRESSURE 被当作 PRESS。"""
    if ocr_textlist is None:
        return []
    values = [ocr_textlist] if isinstance(ocr_textlist, str) else ocr_textlist
    return [token for value in values for token in re.findall(r'[^\W_]+', str(value).upper(), re.UNICODE)]


def _english_phrase_match(ocr_textlist: Iterable[str] | str | None) -> bool:
    """匹配 PRESS ... TO ... DISENGAGE，允许末尾被 OCR 截断。"""
    tokens = _ocr_tokens(ocr_textlist)
    final_prefixes = {'DISE', 'DISEN', 'DISENG', 'DISENGA', 'DISENGAG', 'DISENGAGE'}
    for index, token in enumerate(tokens):
        if token != 'PRESS':
            continue
        to_index = next((pos for pos in range(index + 1, min(index + 4, len(tokens)))
                         if tokens[pos] == 'TO'), None)
        if to_index is None:
            continue
        if any(tokens[pos] in final_prefixes for pos in range(to_index + 1, len(tokens))):
            return True
    return False


def _english_prompt_anchors(ocr_textlist: Iterable[str] | str | None) -> bool:
    """检查英文提示的完整词锚点，避免 PRESSURE/DISTANCE 等子串误触发。"""
    tokens = _ocr_tokens(ocr_textlist)
    final_prefixes = {'DIS', 'DISE', 'DISEN', 'DISENG', 'DISENGA', 'DISENGAG', 'DISENGAGE'}
    return ('PRESS' in tokens and 'TO' in tokens
            and any(token in final_prefixes for token in tokens))


def evaluate_disengage_text(ocr_textlist: Iterable[str] | str | None,
                            similarity: float,
                            threshold: float,
                            target_text: str = 'PRESS TO DISENGAGE') -> DisengageDecision:
    """判断 OCR 是否足以触发脱离；任意相似度不能单独触发。"""
    normalized = normalize_ocr_text(ocr_textlist)
    # 这是专用提示区域，PRESS + TO + DIS(E...) 足以覆盖常见的尾部截断。
    target_normalized = normalize_ocr_text(target_text)
    english_target = target_normalized == 'PRESSTODISENGAGE'
    phrase_match = english_target and _english_phrase_match(ocr_textlist)
    has_prompt_anchors = _english_prompt_anchors(ocr_textlist)
    similarity_match = bool(target_normalized) and ((has_prompt_anchors and similarity > threshold)
                                                     if english_target else similarity > threshold)
    if phrase_match:
        reason = 'phrase'
    elif similarity_match:
        reason = 'similarity'
    else:
        reason = 'no-match'
    return DisengageDecision(normalized, similarity, phrase_match or similarity_match, reason)


def can_trigger_disengage(gui_focus: int, sco_active: bool, cockpit_focus: int = 0) -> bool:
    """只有驾驶舱视角且未处于 SCO 加力状态时才允许发送脱离按键。"""
    return gui_focus == cockpit_focus and not sco_active
