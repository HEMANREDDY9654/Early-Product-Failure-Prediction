"""Tests for aspect_extraction module."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aspect_extraction import (
    extract_aspects,
    extract_aspects_batch,
    get_supported_aspects,
    has_complaint_indicators,
)


class TestExtractAspects:
    """Test aspect extraction for each supported category."""

    def test_battery_aspect(self):
        assert "Battery" in extract_aspects("The battery drains quickly")
        assert "Battery" in extract_aspects("Battery life is terrible")
        assert "Battery" in extract_aspects("power drain is a big issue")

    def test_camera_aspect(self):
        assert "Camera" in extract_aspects("Camera quality is poor")
        assert "Camera" in extract_aspects("Photos are blurry and dark")
        assert "Camera" in extract_aspects("The selfie camera is bad")

    def test_display_aspect(self):
        assert "Display" in extract_aspects("Screen flickers randomly")
        assert "Display" in extract_aspects("Dead pixel appeared on the display")
        assert "Display" in extract_aspects("ghost touch is annoying")

    def test_performance_aspect(self):
        assert "Performance" in extract_aspects("Phone is very slow and laggy")
        assert "Performance" in extract_aspects("The device keeps freezing")
        assert "Performance" in extract_aspects("Performance has degraded")

    def test_charging_aspect(self):
        assert "Charging" in extract_aspects("Slow charging takes forever")
        assert "Charging" in extract_aspects("The charger stopped working")
        assert "Charging" in extract_aspects("USB cable is loose")

    def test_heating_aspect(self):
        assert "Heating" in extract_aspects("Phone overheats during use")
        assert "Heating" in extract_aspects("Gets too hot to hold")
        assert "Heating" in extract_aspects("Thermal throttling is bad")

    def test_software_aspect(self):
        assert "Software" in extract_aspects("Software is buggy")
        assert "Software" in extract_aspects("Frequent crashes after update")
        assert "Software" in extract_aspects("Too many bugs in the firmware")

    def test_durability_aspect(self):
        assert "Durability" in extract_aspects("Build quality is poor and flimsy")
        assert "Durability" in extract_aspects("Phone broke after a small drop")

    def test_connectivity_aspect(self):
        assert "Connectivity" in extract_aspects("WiFi keeps disconnecting")
        assert "Connectivity" in extract_aspects("Bluetooth drops connection")
        assert "Connectivity" in extract_aspects("Weak signal in my area")

    def test_multiple_aspects(self):
        """Test that a review mentioning multiple aspects returns all of them."""
        aspects = extract_aspects("Battery drains fast and the phone overheats when charging")
        assert "Battery" in aspects
        assert "Heating" in aspects
        assert "Charging" in aspects

    def test_empty_text(self):
        assert extract_aspects("") == []
        assert extract_aspects(None) == []

    def test_no_aspect_match(self):
        aspects = extract_aspects("I bought this yesterday.")
        assert aspects == []

    def test_batch_extraction(self):
        texts = [
            "Battery is bad",
            "Camera works great",
            "No issues at all",
        ]
        results = extract_aspects_batch(texts)
        assert len(results) == 3
        assert "Battery" in results[0]
        assert "Camera" in results[1]
        assert results[2] == []


class TestComplaintIndicators:
    """Test complaint/negative indicators detection."""

    def test_complaint_detected(self):
        assert has_complaint_indicators("This is a terrible product")
        assert has_complaint_indicators("Phone stopped working after a week")
        assert has_complaint_indicators("Don't buy this, total waste of money")

    def test_no_complaint(self):
        assert not has_complaint_indicators("Great product, love it!")
        assert not has_complaint_indicators("Works perfectly fine")

    def test_empty_text(self):
        assert not has_complaint_indicators("")
        assert not has_complaint_indicators(None)


class TestSupportedAspects:
    """Test aspect configuration."""

    def test_all_aspects_present(self):
        aspects = get_supported_aspects()
        expected = [
            "Battery", "Camera", "Display", "Performance",
            "Charging", "Heating", "Software", "Durability", "Connectivity",
        ]
        for asp in expected:
            assert asp in aspects
