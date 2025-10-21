import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.config_reader import get_table_enable
from mineru.utils.enum_class import ContentType
from mineru.utils.pdf_image_tools import get_crop_img


def _should_enable_recognition() -> bool:
    env_flag = os.getenv("MINERU_VLM_TABLE_RECOGNITION_ENABLE", "true").lower() == "true"
    return env_flag and get_table_enable(True)


@dataclass
class _RecognitionResult:
    html: Optional[str]
    source: str


class _VLMTableRecognitionManager:
    """Lazy singleton responsible for upgrading VLM table spans with structural HTML."""

    _instance: "_VLMTableRecognitionManager" | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._enabled = _should_enable_recognition()
        self._atom_manager = AtomModelSingleton()
        self._table_cls_model = None
        self._wireless_table_model = None
        self._wired_table_model = None
        self._orientation_model = None
        # 默认使用轻量中文模型，与原 pipeline 初始化保持一致
        self._lang = os.getenv("MINERU_VLM_TABLE_LANG", "ch_lite")

    def enhance_table_span(self, span: dict, page_pil_img: Image.Image, scale: float) -> None:
        if not self._enabled:
            return
        if span.get("type") != ContentType.TABLE:
            return

        result = self._recognize_table(span, page_pil_img, scale)
        if result and result.html:
            span["html"] = result.html
            span["table_recognition_source"] = result.source

    def _recognize_table(self, span: dict, page_pil_img: Image.Image, scale: float) -> Optional[_RecognitionResult]:
        try:
            table_img = self._crop_table_image(span, page_pil_img, scale)
            if table_img is None:
                return None

            table_np = np.asarray(table_img)
            if table_np.size == 0:
                return None

            table_np = self._apply_orientation(table_np)
            table_img = Image.fromarray(table_np)

            wireless_html = self._run_wireless(table_img)
            final_html = wireless_html
            source = "wireless"

            # 根据分类器结果决定是否使用 wired 推理
            label, _ = self._get_table_classifier().predict(table_img)
            if label == AtomicModel.WiredTable:
                wired_html = self._run_wired(table_img, wireless_html)
                if wired_html:
                    final_html = wired_html
                    source = "wired"

            if not final_html:
                return None

            return _RecognitionResult(html=final_html, source=source)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to enhance VLM table span: {}", exc,
            )
            return None

    def _crop_table_image(
        self, span: dict, page_pil_img: Image.Image, scale: float,
    ) -> Optional[Image.Image]:
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        crop_img = get_crop_img(tuple(bbox), page_pil_img, scale=scale)
        if crop_img.mode != "RGB":
            crop_img = crop_img.convert("RGB")
        width, height = crop_img.size
        if width < 8 or height < 8:
            return None
        return crop_img

    def _apply_orientation(self, table_np: np.ndarray) -> np.ndarray:
        orientation_model = self._get_orientation_model()
        if orientation_model is None:
            return table_np
        label = orientation_model.predict(table_np)
        if label == "270":
            return np.rot90(table_np, k=3)
        if label == "90":
            return np.rot90(table_np, k=1)
        if label == "180":
            return np.rot90(table_np, k=2)
        return table_np

    def _run_wireless(self, table_img: Image.Image) -> Optional[str]:
        wireless_model = self._get_wireless_table_model()
        if wireless_model is None:
            return None
        html_code, *_ = wireless_model.predict(table_img)
        if isinstance(html_code, list):
            html_code = html_code[0] if html_code else None
        return html_code

    def _run_wired(self, table_img: Image.Image, wireless_html: Optional[str]) -> Optional[str]:
        wired_model = self._get_wired_table_model()
        if wired_model is None:
            return None
        return wired_model.predict(table_img, None, wireless_html or "")

    def _get_table_classifier(self):
        if self._table_cls_model is None:
            self._table_cls_model = self._atom_manager.get_atom_model(
                atom_model_name=AtomicModel.TableCls,
            )
        return self._table_cls_model

    def _get_wireless_table_model(self):
        if self._wireless_table_model is None:
            self._wireless_table_model = self._atom_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
                lang=self._lang,
            )
        return self._wireless_table_model

    def _get_wired_table_model(self):
        if self._wired_table_model is None:
            self._wired_table_model = self._atom_manager.get_atom_model(
                atom_model_name=AtomicModel.WiredTable,
                lang=self._lang,
            )
        return self._wired_table_model

    def _get_orientation_model(self):
        if self._orientation_model is None:
            try:
                self._orientation_model = self._atom_manager.get_atom_model(
                    atom_model_name=AtomicModel.ImgOrientationCls,
                    lang=self._lang,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Orientation model unavailable: {}", exc)
                self._orientation_model = None
        return self._orientation_model


def enhance_table_span(span: dict, page_pil_img: Image.Image, scale: float) -> None:
    """Public helper consumed by VLM pipeline to upgrade table HTML in-place."""
    manager = _VLMTableRecognitionManager()
    manager.enhance_table_span(span, page_pil_img, scale)
