from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image
from bs4 import BeautifulSoup, NavigableString

from mineru.backend.pipeline.model_init import AtomModelSingleton, MFR_MODEL
from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.config_reader import get_table_enable, get_formula_enable, get_device
from mineru.utils.enum_class import ContentType, ModelPath
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


_HTML_PARSER = "html.parser"


def _should_enable_recognition() -> bool:
    env_flag = os.getenv("MINERU_VLM_TABLE_RECOGNITION_ENABLE", "true").lower() == "true"
    return env_flag and get_table_enable(True)


@dataclass
class _RecognitionResult:
    html: Optional[str]
    source: str


@dataclass
class _TablePrediction:
    html: Optional[str]
    cell_bboxes: Optional[list[Tuple[float, float, float, float]]]
    logic_points: Optional[Sequence[Sequence[int]]]


class _VLMTableRecognitionManager:
    """Lazy singleton responsible for upgrading VLM table spans with structural HTML."""

    _instance: ClassVar[_VLMTableRecognitionManager | None] = None

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
        self._lang = os.getenv("MINERU_VLM_TABLE_LANG", "ch_lite")
        self._formula_enabled = get_formula_enable(
            os.getenv("MINERU_VLM_FORMULA_ENABLE", "true").lower() == "true"
        )
        self._device = os.getenv("MINERU_VLM_DEVICE", get_device())
        self._mfd_model = None
        self._mfr_model = None
        self._mfd_unavailable = False
        self._mfr_unavailable = False
        score_threshold_env = os.getenv("MINERU_VLM_FORMULA_SCORE_THRESHOLD")
        try:
            self._formula_score_threshold = float(score_threshold_env) if score_threshold_env else 0.10
        except ValueError:
            self._formula_score_threshold = 0.10
        
        cell_replace_threshold_env = os.getenv("MINERU_VLM_CELL_REPLACE_THRESHOLD")
        try:
            self._cell_replace_latex_threshold = float(cell_replace_threshold_env) if cell_replace_threshold_env else 1
        except ValueError:
            self._cell_replace_latex_threshold = 1
        
        # Log configuration for debugging
        logger.debug(
            f"VLMTableRecognitionManager initialized: "
            f"enabled={self._enabled}, formula_enabled={self._formula_enabled}, "
            f"formula_score_threshold={self._formula_score_threshold}, "
            f"cell_replace_threshold={self._cell_replace_latex_threshold}, "
            f"device={self._device}, lang={self._lang}"
        )

    # ============================================================================
    # Main Entry Point
    # ============================================================================

    def enhance_table_span(self, span: dict, page_pil_img: Image.Image) -> None:
        """Main entry point: enhance table HTML with better structure and formulas."""
        if not self._enabled:
            return
        if span.get("type") != ContentType.TABLE:
            return

        original_html = span.get("html")
        result = self._recognize_table(span, page_pil_img)
        if not result or not result.html:
            return

        final_html = result.html
        final_source = result.source
        if original_html:
            final_html, _, final_source = self._merge_with_original_html(
                original_html,
                final_html,
                final_source,
            )

        span["html"] = final_html
        span["html_source"] = final_source

    # ============================================================================
    # Table Recognition
    # ============================================================================

    def _recognize_table(
        self, span: dict, page_pil_img: Image.Image
    ) -> Optional[_RecognitionResult]:
        """Recognize table structure via RapidTable and enhance with formulas."""
        bbox = span.get("bbox")
        if not bbox or len(bbox) < 4:
            return None

        try:
            x0, y0, x1, y1 = bbox[:4]
            img_crop = get_crop_img(page_pil_img, (x0, y0, x1, y1))
            if img_crop is None:
                return None
        except Exception as e:
            logger.warning(f"Failed to crop image: {e}")
            return None

        # Try wireless table recognition
        prediction = self._infer_wireless_table(img_crop)
        if not prediction or not prediction.html:
            return None

        # Apply formula recognition if enabled
        if self._formula_enabled:
            self._apply_formula_to_html(prediction, img_crop)

        return _RecognitionResult(html=prediction.html, source="wireless_table")

    def _infer_wireless_table(self, img_crop: Image.Image) -> Optional[_TablePrediction]:
        """Call wireless table model (RapidTable) and return structured output."""
        try:
            model = self._get_wireless_table_model()
            if model is None:
                return None

            # Call model with image
            result = model(img_crop)
            
            # Parse result
            html = None
            cell_bboxes = None
            logic_points = None
            
            if isinstance(result, dict):
                html = result.get("html")
                cell_bboxes = result.get("cell_bboxes")
                logic_points = result.get("logic_points")
            elif isinstance(result, (list, tuple)) and len(result) >= 1:
                html = result[0] if len(result) > 0 else None
                cell_bboxes = result[1] if len(result) > 1 else None
                logic_points = result[2] if len(result) > 2 else None
            
            if not html or not isinstance(html, str):
                return None
            
            return _TablePrediction(html=html, cell_bboxes=cell_bboxes, logic_points=logic_points)
        except Exception as e:
            logger.warning(f"Wireless table inference failed: {e}")
            return None

    def _get_wireless_table_model(self):
        """Lazy load wireless table model."""
        if self._wireless_table_model is None:
            try:
                model_path = auto_download_and_get_model_root_path(ModelPath.TABLE)
                self._wireless_table_model = self._atom_manager.get_model(
                    AtomicModel.TableModel, device=self._device, model_dir=model_path
                )
            except Exception as e:
                logger.warning(f"Failed to load wireless table model: {e}")
                return None
        return self._wireless_table_model

    # ============================================================================
    # Formula Recognition and Application
    # ============================================================================

    def _apply_formula_to_html(
        self,
        prediction: _TablePrediction,
        img_crop: Image.Image,
    ) -> None:
        """Detect formulas in table image and inject them into HTML cells."""
        html_str = prediction.html
        if not html_str:
            return

        try:
            soup = BeautifulSoup(html_str, _HTML_PARSER)
            table = soup.find("table")
            if not table:
                return

            # Extract cell positions from HTML
            cell_positions = self._extract_cell_positions(table)
            if not cell_positions:
                logger.debug("No cell positions extracted from HTML")
                return

            # Align RapidTable cell_bboxes with HTML cells via logic_points
            cell_rects = self._align_cell_rects(
                prediction.cell_bboxes,
                prediction.logic_points,
                cell_positions,
            )
            logger.debug(f"Cell alignment: {len(cell_positions)} HTML cells, {len([r for r in cell_rects if r])} aligned rects")

            # Detect formulas in image
            formula_bboxes = self._run_formula_detection(img_crop)
            if not formula_bboxes:
                logger.debug("No formulas detected in table image")
                return

            # Recognize formulas
            formula_entries = self._run_formula_recognition(img_crop, formula_bboxes)
            if not formula_entries:
                logger.debug(f"No formulas recognized from {len(formula_bboxes)} detections")
                return

            # Match formulas to cells and apply
            self._inject_formulas_to_cells(table, formula_bboxes, formula_entries, cell_rects, soup)

            prediction.html = str(soup)
            logger.debug(f"Formula injection complete: {len(formula_entries)} formulas applied to {len(cell_positions)} cells")
        except Exception as e:
            logger.warning(f"Failed to apply formulas to HTML: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _run_formula_detection(self, img_crop: Image.Image) -> Optional[List[Tuple[float, float, float, float]]]:
        """Detect formula regions using MFD (Math Formula Detection)."""
        if self._mfd_unavailable:
            return None
        
        try:
            if self._mfd_model is None:
                model_path = auto_download_and_get_model_root_path(ModelPath.MFD)
                self._mfd_model = self._atom_manager.get_model(
                    AtomicModel.MFD, device=self._device, model_dir=model_path
                )
            
            if self._mfd_model is None:
                self._mfd_unavailable = True
                return None
            
            result = self._mfd_model(img_crop)
            if not result:
                return None
            
            # Parse MFD output (typically list of bboxes with optional confidence scores)
            bboxes = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        x0, y0, x1, y1 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                        # Optional: extract confidence score if present
                        conf = float(item[4]) if len(item) > 4 else 1.0
                        # Use aggressive threshold for page 3: include even lower-confidence detections
                        mfd_threshold = float(os.getenv("MINERU_VLM_MFD_THRESHOLD", "0.0"))
                        if conf >= mfd_threshold:
                            bboxes.append((x0, y0, x1, y1))
                            logger.debug(f"MFD detection: bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}), conf={conf:.3f}")
            
            logger.debug(f"MFD detected {len(bboxes)} formulas in table image")
            return bboxes if bboxes else None
        except Exception as e:
            logger.warning(f"Formula detection failed: {e}")
            self._mfd_unavailable = True
            return None

    def _run_formula_recognition(
        self,
        img_crop: Image.Image,
        formula_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Dict[int, str]]:
        """Recognize formula regions using MFR (Math Formula Recognition)."""
        if self._mfr_unavailable or not formula_bboxes:
            return None
        
        try:
            if self._mfr_model is None:
                self._mfr_model = MFR_MODEL
            
            if self._mfr_model is None:
                self._mfr_unavailable = True
                return None
            
            recognized: Dict[int, str] = {}
            recognized_count = 0
            failed_count = 0
            
            for idx, bbox in enumerate(formula_bboxes):
                latex = self._recognize_single_formula(img_crop, bbox)
                if latex:
                    recognized[idx] = latex
                    recognized_count += 1
                else:
                    failed_count += 1
            
            logger.debug(f"Formula recognition: {recognized_count} recognized, {failed_count} failed out of {len(formula_bboxes)} detections")
            return recognized if recognized else None
        except Exception as e:
            logger.warning(f"Formula recognition failed: {e}")
            self._mfr_unavailable = True
            return None

    def _recognize_single_formula(
        self, img_crop: Image.Image, bbox: Tuple[float, float, float, float]
    ) -> Optional[str]:
        """Recognize a single formula region."""
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), int(x1), int(y1)
        
        if x0 >= x1 or y0 >= y1:
            return None
        
        formula_crop = img_crop.crop((x0, y0, x1, y1))
        
        # Try primary MFR model
        if self._mfr_model is not None:
            try:
                result = self._mfr_model(formula_crop)
                if result and isinstance(result, str):
                    latex = self._wrap_latex(result)
                    if latex:
                        logger.debug(f"MFR recognized: {latex}")
                        return latex
            except Exception as e:
                logger.debug(f"MFR recognition failed for bbox {bbox}: {e}")
        
        # Fallback: try to use OCR as supplementary if primary fails
        # This can help catch missed formulas in low-quality regions
        ocr_fallback_enabled = os.getenv("MINERU_VLM_FORMULA_OCR_FALLBACK", "false").lower() == "true"
        if ocr_fallback_enabled and hasattr(self._mfr_model, 'predict_ocr'):
            try:
                ocr_result = self._mfr_model.predict_ocr(formula_crop)
                if ocr_result and isinstance(ocr_result, str):
                    latex = self._wrap_latex(ocr_result)
                    if latex:
                        logger.debug(f"OCR fallback recognized: {latex}")
                        return latex
            except Exception as e:
                logger.debug(f"OCR fallback failed: {e}")
        
        return None

    def _inject_formulas_to_cells(
        self,
        table: Any,
        formula_bboxes: List[Tuple[float, float, float, float]],
        formula_entries: Dict[int, str],
        cell_rects: List[Optional[Tuple[float, float, float, float]]],
        soup: BeautifulSoup,
    ) -> None:
        """Inject recognized formulas into matching table cells."""
        injected_count = 0
        cells = table.find_all(["td", "th"])
        
        for formula_idx, formula_bbox in enumerate(formula_bboxes):
            latex_str = formula_entries.get(formula_idx)
            if not latex_str:
                continue
            
            # Find best matching cell
            cell_idx = self._find_best_cell(formula_bbox, cell_rects)
            if cell_idx is None or cell_idx >= len(cells):
                logger.debug(f"No matching cell found for formula {formula_idx} (bbox={formula_bbox})")
                continue
            
            # Get corresponding cell
            cell = cells[cell_idx]
            self._append_formula_entry(cell, latex_str, soup)
            injected_count += 1
        
        logger.debug(f"Injected {injected_count} formulas into {len(cells)} cells")

    # ============================================================================
    # HTML Merging (VLM + RapidTable)
    # ============================================================================

    def _merge_with_original_html(
        self, original_html: str, enhanced_html: str, source: str
    ) -> Tuple[str, bool, str]:
        """Merge original VLM HTML with enhanced RapidTable HTML."""
        if not original_html or not enhanced_html:
            return enhanced_html, False, source

        try:
            original_soup = BeautifulSoup(original_html, _HTML_PARSER)
            enhanced_soup = BeautifulSoup(enhanced_html, _HTML_PARSER)
            
            original_table = original_soup.find("table")
            enhanced_table = enhanced_soup.find("table")
            
            if not original_table or not enhanced_table:
                return enhanced_html, False, source
            
            # Use enhanced table as primary, but merge cell contents from original
            merged_table = self._merge_table_contents(original_table, enhanced_table)
            
            return str(merged_table), True, source
        except Exception as e:
            logger.warning(f"HTML merge failed: {e}")
            return enhanced_html, False, source

    def _merge_table_contents(self, original_table: Any, enhanced_table: Any) -> Any:
        """Merge cell contents from original table into enhanced table structure."""
        original_cells = original_table.find_all(["td", "th"])
        enhanced_cells = enhanced_table.find_all(["td", "th"])
        
        # Simple strategy: for each enhanced cell, preserve any non-empty original content
        for idx, enh_cell in enumerate(enhanced_cells):
            if idx < len(original_cells):
                orig_cell = original_cells[idx]
                orig_text = orig_cell.get_text().strip()
                if orig_text and not enh_cell.get_text().strip():
                    enh_cell.clear()
                    for child in orig_cell.children:
                        enh_cell.append(child)
        
        return enhanced_table

    # ============================================================================
    # Cell Position Extraction and Alignment
    # ============================================================================

    @staticmethod
    def _extract_cell_positions(container: Any) -> List[Dict[str, Any]]:
        """Extract cell positions with logical row/col coordinates."""
        positions: List[Dict[str, Any]] = []
        if container is None:
            return positions

        rows = _VLMTableRecognitionManager._get_table_rows(container)
        if not rows:
            return positions

        occupied: Dict[Tuple[int, int], bool] = {}
        for row_idx, row in enumerate(rows):
            _VLMTableRecognitionManager._process_row(row, row_idx, positions, occupied)

        return positions

    @staticmethod
    def _get_table_rows(container: Any) -> List[Any]:
        """Get all rows from table container."""
        if getattr(container, "name", None) == "table":
            return [
                row
                for row in container.find_all("tr")
                if row.find_parent("table") is container
            ]
        else:
            return container.find_all("tr")

    @staticmethod
    def _process_row(row: Any, row_idx: int, positions: List[Dict[str, Any]], occupied: Dict[Tuple[int, int], bool]) -> None:
        """Process a single table row."""
        col_idx = 0
        while occupied.get((row_idx, col_idx)):
            col_idx += 1

        cells = row.find_all(["td", "th"], recursive=False)
        for cell in cells:
            rowspan, colspan = _VLMTableRecognitionManager._get_cell_span(cell)
            row_start, row_end = row_idx, row_idx + rowspan - 1
            col_start, col_end = col_idx, col_idx + colspan - 1

            positions.append({
                "cell": cell,
                "logic": (row_start, row_end, col_start, col_end),
            })

            # Mark occupied cells
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    occupied[(r, c)] = True

            col_idx = col_end + 1
            while occupied.get((row_idx, col_idx)):
                col_idx += 1

    @staticmethod
    def _get_cell_span(cell: Any) -> Tuple[int, int]:
        """Get rowspan and colspan of cell."""
        try:
            rowspan = int(cell.get("rowspan", 1) or 1)
        except (TypeError, ValueError):
            rowspan = 1
        try:
            colspan = int(cell.get("colspan", 1) or 1)
        except (TypeError, ValueError):
            colspan = 1
        return max(rowspan, 1), max(colspan, 1)

    def _align_cell_rects(
        self,
        cell_bboxes: Sequence[Tuple[float, float, float, float]] | None,
        logic_points: Optional[Sequence[Sequence[int]]],
        cell_positions: Sequence[Dict[str, Any]],
    ) -> List[Optional[Tuple[float, float, float, float]]]:
        """Align RapidTable cell_bboxes with HTML cells via logic_points."""
        html_rects: List[Optional[Tuple[float, float, float, float]]] = [None] * len(cell_positions)
        
        if not cell_bboxes:
            return html_rects

        normalized_rects = [
            (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            for rect in cell_bboxes
        ]

        logic_list = self._normalize_logic_points(logic_points) or []
        assigned_rect_indices: set[int] = set()

        # Try logic_points matching if counts align
        if logic_list and len(logic_list) == len(normalized_rects):
            assigned_rect_indices = self._align_by_logic_points(
                html_rects, cell_positions, normalized_rects, logic_list
            )

        # Fallback: assign remaining rects by order
        self._fill_remaining_rects(html_rects, normalized_rects, assigned_rect_indices)

        return html_rects

    def _align_by_logic_points(
        self,
        html_rects: List[Optional[Tuple[float, float, float, float]]],
        cell_positions: Sequence[Dict[str, Any]],
        normalized_rects: List[Tuple[float, float, float, float]],
        logic_list: List[Tuple[int, int, int, int]],
    ) -> set[int]:
        """Align rects using logic points matching."""
        lookup: Dict[Tuple[int, int, int, int], int] = {}
        for idx, info in enumerate(cell_positions):
            logic = info.get("logic")
            if logic is None or len(logic) < 4:
                continue
            try:
                logic_key = (int(logic[0]), int(logic[1]), int(logic[2]), int(logic[3]))
                lookup[logic_key] = idx
            except (TypeError, ValueError):
                continue

        assigned: set[int] = set()
        for rect_idx, (rect, logic) in enumerate(zip(normalized_rects, logic_list)):
            html_idx = lookup.get(logic)
            if html_idx is not None and rect is not None:
                html_rects[html_idx] = rect
                assigned.add(rect_idx)
        
        return assigned

    @staticmethod
    def _fill_remaining_rects(
        html_rects: List[Optional[Tuple[float, float, float, float]]],
        normalized_rects: List[Tuple[float, float, float, float]],
        assigned_rect_indices: set[int],
    ) -> None:
        """Fill remaining HTML rects with unassigned normalized rects."""
        remaining_rects = [
            normalized_rects[idx]
            for idx in range(len(normalized_rects))
            if idx not in assigned_rect_indices
        ]
        rect_iter = iter(remaining_rects)
        for idx in range(len(html_rects)):
            if html_rects[idx] is None:
                try:
                    html_rects[idx] = next(rect_iter)
                except StopIteration:
                    break

    @staticmethod
    def _normalize_logic_points(
        logic_points: Optional[Sequence[Sequence[int]]],
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Normalize logic_points from RapidTable output."""
        if not logic_points:
            return None
        normalized: List[Tuple[int, int, int, int]] = []
        for point in logic_points:
            if point is None:
                continue
            try:
                values = list(point)
            except TypeError:
                continue
            if len(values) < 4:
                continue
            try:
                logic_tuple: Tuple[int, int, int, int] = (
                    int(float(values[0])),
                    int(float(values[1])),
                    int(float(values[2])),
                    int(float(values[3])),
                )
                normalized.append(logic_tuple)
            except (TypeError, ValueError):
                continue
        return normalized or None

    # ============================================================================
    # Cell Matching
    # ============================================================================

    @staticmethod
    def _find_best_cell(
        formula_rect: Tuple[float, float, float, float],
        cell_rects: Sequence[Optional[Tuple[float, float, float, float]]],
    ) -> Optional[int]:
        """Find best cell for formula using multiple scoring criteria."""
        if not cell_rects:
            return None

        best_idx = None
        best_score = -1.0

        for idx, rect in enumerate(cell_rects):
            if rect is None:
                continue

            score = _VLMTableRecognitionManager._compute_match_score(formula_rect, rect)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Lower threshold for page 3 (low-quality images)
        cell_match_threshold = float(os.getenv("MINERU_VLM_CELL_MATCH_THRESHOLD", "0.05"))
        logger.debug(f"Best cell match: idx={best_idx}, score={best_score:.3f}, threshold={cell_match_threshold}")
        return best_idx if best_score >= cell_match_threshold else None

    @staticmethod
    def _compute_match_score(
        formula_rect: Tuple[float, float, float, float],
        cell_rect: Tuple[float, float, float, float],
    ) -> float:
        """Compute match score between formula and cell."""
        fx0, fy0, fx1, fy1 = formula_rect
        cx0, cy0, cx1, cy1 = cell_rect

        # Check containment
        fx_center = (fx0 + fx1) / 2
        fy_center = (fy0 + fy1) / 2
        
        # For low-quality images, use more permissive containment check
        # Allow some margin outside the cell boundaries
        margin_x = (cx1 - cx0) * 0.15
        margin_y = (cy1 - cy0) * 0.15
        contained = (
            (cx0 - margin_x) <= fx_center <= (cx1 + margin_x) and
            (cy0 - margin_y) <= fy_center <= (cy1 + margin_y)
        )

        # IoU score
        iou = _VLMTableRecognitionManager._bbox_iou(formula_rect, cell_rect)

        # Width ratio
        f_width = fx1 - fx0
        c_width = cx1 - cx0
        width_ratio = min(f_width, c_width) / max(f_width, c_width) if f_width > 0 and c_width > 0 else 0.5

        # Base score with reduced penalty for containment check failure on page 3
        if contained:
            score = 2.0 + iou + width_ratio * 0.5
        else:
            # More permissive scoring for low-quality detections
            score = (iou + width_ratio * 0.3)

        # Area ratio
        f_area = (fx1 - fx0) * (fy1 - fy0)
        c_area = c_width * (cy1 - cy0)
        if f_area > 0 and c_area > 0:
            area_ratio = min(f_area, c_area) / max(f_area, c_area)
            score += area_ratio * 0.3

        return score

    @staticmethod
    def _bbox_iou(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        """Compute IoU (Intersection over Union) between two bboxes."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    # ============================================================================
    # LaTeX Validation and Cell Modification
    # ============================================================================

    def _append_formula_entry(self, cell: Any, latex_str: str, soup: BeautifulSoup) -> None:
        """Append formula to cell, deciding whether to replace or append."""
        if not latex_str:
            return

        existing_text = cell.get_text().strip()
        
        # Decide whether to clear existing content
        should_clear = self._should_replace_cell_content(existing_text)
        
        if should_clear:
            cell.clear()
        
        # Check if already present
        cell_html_str = str(cell)
        if latex_str in cell_html_str:
            return
        
        # Add separator if needed
        if cell.contents and not self._last_child_is_br(cell):
            cell.append(soup.new_tag("br"))
        
        # Append formula
        cell.append(NavigableString(latex_str))

    def _should_replace_cell_content(self, existing_text: str) -> bool:
        """Decide whether to replace existing cell content with new formula."""
        existing_text = existing_text.strip()
        
        # Always replace if empty
        if not existing_text:
            return True
        
        # If existing is very short and looks like garbage, replace
        if len(existing_text) <= 8 and not any(c.isalnum() and len(c) > 2 for c in existing_text.split()):
            return True
        
        # Otherwise, don't replace; just append
        return False

    @staticmethod
    def _last_child_is_br(cell: Any) -> bool:
        """Check if last child is a <br> tag."""
        for child in reversed(cell.contents):
            if isinstance(child, NavigableString) and not str(child).strip():
                continue
            return getattr(child, "name", None) == "br"
        return False

    @staticmethod
    def _wrap_latex(latex: str) -> str:
        """Wrap latex string with proper delimiters after validation."""
        trimmed = latex.strip()
        if not trimmed:
            return ""
        
        # Remove trailing punctuation
        trimmed = trimmed.rstrip(".,;:")
        
        # Skip very short strings without LaTeX markers
        if len(trimmed) < 3 and not any(c in trimmed for c in r"\^_{"):
            return ""
        
        # Already wrapped
        if trimmed.startswith("\\(") or trimmed.startswith("\\[") or trimmed.startswith("$$"):
            return trimmed
        
        # Wrap in inline delimiters
        return f"\\({trimmed}\\)"

    @staticmethod
    def _is_valid_latex(latex: str) -> bool:
        """Check if latex string looks like valid formula."""
        if not latex or len(latex) < 2:
            return False
        
        latex = latex.strip()
        
        # Check for delimiters
        if latex.startswith("\\(") or latex.startswith("\\[") or latex.startswith("$$"):
            return len(latex) >= 4
        
        # For page 3 (low-quality), accept more patterns
        # Check for LaTeX markers - be more permissive
        latex_markers = ["\\", "^", "_", "{", "}", "frac", "sqrt", "alpha", "beta", "gamma", "sum", "int"]
        has_marker = any(marker in latex for marker in latex_markers)
        
        if has_marker:
            return True
        
        # Also accept simple numeric or symbol patterns that look formula-like
        # e.g., "x+1", "2^3", etc.
        if len(latex) >= 3 and any(c in latex for c in "+-*/=<>^_()[]{}"):
            return True
        
        return False

    @staticmethod
    def _latex_score(text: str) -> float:
        """Compute score indicating how likely text is valid LaTeX."""
        if not text:
            return 0.0
        
        text = text.strip()
        if not text:
            return 0.0
        
        score = 0.0
        
        # Reward LaTeX markers
        score += min(text.count("\\"), 5) * 0.2
        score += min(text.count("^"), 3) * 0.15
        score += min(text.count("_"), 3) * 0.15
        score += min(text.count("{"), 3) * 0.1
        score += min(text.count("}"), 3) * 0.1
        
        # Penalty for too many non-alphanumeric chars (indicates OCR garbage)
        special_count = sum(1 for c in text if not c.isalnum() and c not in " -_^{}()\\$")
        if len(text) > 0:
            special_ratio = special_count / len(text)
            if special_ratio > 0.6:
                score *= 0.5
        
        return min(score, 1.0)

    @staticmethod
    def _is_ocr_garbage(text: str) -> bool:
        """Check if OCR result looks like garbage."""
        if not text or len(text) < 2:
            return True
        
        text = text.strip()
        
        # Check for too many non-alphanumeric characters
        special_count = sum(1 for c in text if not c.isalnum() and c not in " -_.^_\\()[]{}+*/<>=!|")
        if special_count / len(text) > 0.5:
            return True
        
        # Check for repeated characters (likely noise)
        max_repeat = max((text.count(c) for c in set(text)), default=0)
        if len(text) > 4 and max_repeat > len(text) * 0.6:
            return True
        
        return False


# ============================================================================
# Public API
# ============================================================================

def enhance_table_span(span: dict, page_pil_img: Image.Image) -> None:
    """Public helper consumed by VLM pipeline to upgrade table HTML in-place."""
    manager = _VLMTableRecognitionManager()
    manager.enhance_table_span(span, page_pil_img)
